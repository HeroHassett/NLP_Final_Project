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
        print(f"Error loading label mappings: {e}")
        idx2label, label2idx = {}, {}

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
        examples["tokens"], is_split_into_words=True, truncation=True, padding=True
    )
    # Need to add usage metrics here

    all_labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs


metric = evaluate.load("seqeval")


def compute_metrics(p, idx2label: dict):
    predictions, labels_batch = p
    predictions = np.argmax(predictions, axis=-1)

    labels_list = [idx2label[i] for i in range(len(idx2label))]
    true_labels = []
    true_predictions = []

    for l, p_ in zip(labels_batch, predictions):
        filtered_labels = [labels_list[i] for i, lab in enumerate(l) if lab != -100]
        filtered_preds = [labels_list[i] for i, lab in enumerate(l) if lab != -100]
        true_labels.append(filtered_labels)
        true_predictions.append(filtered_preds)

    results = metric.compute(predictions=true_predictions, references=true_labels)

    micro_f1 = results["overall_f1"]

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "micro_f1": micro_f1,
        "accuracy": results["overall_accuracy"],
    }
