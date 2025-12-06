"Utilities for training and testing"

import evaluate
import numpy as np
import requests
from datasets import ClassLabel, Sequence, load_dataset
from transformers import AutoTokenizer


def load_rate_ner_dataset(directory="all", ner_type="IOB"):
    idx2label_url = "https://huggingface.co/datasets/Angelakeke/RaTE-NER/resolve/main/idx2label.json"
    label2idx_url = "https://huggingface.co/datasets/Angelakeke/RaTE-NER/resolve/main/label2idx.json"

    try:
        idx2label = requests.get(idx2label_url).json()
        label2idx = requests.get(label2idx_url).json()
        label_names = [idx2label[str(i)] for i in range(len(idx2label))]
        ner_classlabel = ClassLabel(names=label_names)
        ner_sequence = Sequence(feature=ner_classlabel)
    except Exception as e:
        raise e

    data_files = {
        "train": f"{directory}/train_{ner_type}.json",
        "validation": f"{directory}/dev_{ner_type}.json",
        "test": f"{directory}/test_{ner_type}.json",
    }

    dataset = load_dataset("Angelakeke/RaTE-NER", data_files=data_files)

    # Convert ner_tags to Sequence(ClassLabel)
    if label2idx and idx2label:
        for split in ["train", "validation", "test"]:
            dataset[split] = dataset[split].cast_column("ner_tags", ner_sequence)

    return dataset, idx2label, label2idx


def tokenize_and_align_labels(examples, tokenizer: AutoTokenizer, label2idx: dict):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=256
    )

    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)

        label_ids = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)

            elif word_idx != previous_word_idx:
                # new word → take label
                label_ids.append(labels[word_idx])

            else:
                # same wordpiece → mask out
                label_ids.append(-100)

            previous_word_idx = word_idx

        # ⚠ Ensure labels exactly match sequence length
        seq_len = len(tokenized_inputs["input_ids"][i])
        if len(label_ids) < seq_len:
            label_ids += [-100] * (seq_len - len(label_ids))

        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs



metric = evaluate.load("seqeval")

def compute_metrics(p, idx2label: dict):
    logits, labels_batch = p
    predictions = np.argmax(logits, axis=-1)

    # id → label string
    id2label_list = [idx2label[str(i)] for i in range(len(idx2label))]

    true_predictions = []
    true_labels = []

    for preds, labs in zip(predictions, labels_batch):
        pred_labels = []
        gold_labels = []

        for p_i, l_i in zip(preds, labs):
            if l_i == -100:
                continue
            gold_labels.append(id2label_list[l_i])
            pred_labels.append(id2label_list[p_i])

        true_predictions.append(pred_labels)
        true_labels.append(gold_labels)

    results = metric.compute(predictions=true_predictions, references=true_labels)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "micro_f1": results["overall_f1"],   # your main metric
        "accuracy": results["overall_accuracy"],
    }

