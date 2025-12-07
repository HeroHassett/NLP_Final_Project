"Utilities for training and testing"

import evaluate
import numpy as np
import requests
import time
import warnings
from pathlib import Path
from datasets import ClassLabel, Sequence, load_dataset
from transformers import AutoTokenizer
from typing import List, Sequence as SeqType, Dict, Any

try:
    import torch
except Exception:  # pragma: no cover - optional dependency at import time
    torch = None

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None

try:
    import pynvml
except Exception:  # pragma: no cover - optional dependency
    pynvml = None


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

    # Prefer local shard files if they exist; otherwise fall back to hosted RaTE-NER
    dir_path = Path(directory)
    candidate_files = {
        "train": dir_path / f"train_{ner_type}.json",
        "validation": dir_path / f"dev_{ner_type}.json",
        "test": dir_path / f"test_{ner_type}.json",
    }
    dataset = None

    if all(p.exists() for p in candidate_files.values()):
        data_files = {split: str(path) for split, path in candidate_files.items()}
        dataset = load_dataset("json", data_files=data_files)

    # Fallback to remote hosted files via HTTPS if local shards are missing
    if dataset is None:
        base_url = "https://huggingface.co/datasets/Angelakeke/RaTE-NER/resolve/main"
        data_files = {
            "train": f"{base_url}/{directory}/train_{ner_type}.json",
            "validation": f"{base_url}/{directory}/dev_{ner_type}.json",
            "test": f"{base_url}/{directory}/test_{ner_type}.json",
        }
        dataset = load_dataset("json", data_files=data_files)

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

def compute_metrics(predictions_obj, idx2label: dict):
    """
    Compute seqeval metrics from a HF Trainer eval prediction or a (logits, labels) tuple.
    Supports both the EvalPrediction object and raw tuple for compatibility.
    """
    if hasattr(predictions_obj, "predictions") and hasattr(predictions_obj, "label_ids"):
        logits = predictions_obj.predictions
        labels_batch = predictions_obj.label_ids
    else:
        logits, labels_batch = predictions_obj

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


def compute_tag_metrics(
    predictions: SeqType[SeqType[str]], references: SeqType[SeqType[str]]
) -> Dict[str, float]:
    """
    Seqeval-based precision/recall/micro_f1 for already-decoded tag lists.
    predictions/references are lists of label strings (IOB format, e.g., \"B-FINDING\").
    """
    results = metric.compute(predictions=predictions, references=references)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "micro_f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


class ResourceTracker:
    """
    Lightweight resource/throughput tracker used during inference or training loops.
    Captures CPU %, RAM, GPU memory, GPU power (if NVML is available), and latency/throughput.
    """

    def __init__(self):
        self.latencies = []
        self.samples = 0
        self.tokens = 0
        self.start_time = time.time()
        self.cpu_samples = []
        self.ram_samples = []
        self.gpu_power_samples = []
        self.max_gpu_mem = 0

        self._nvml_handle = None
        if torch and torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        if pynvml:
            try:
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception as exc:  # pragma: no cover - hardware dependent
                warnings.warn(f"NVML unavailable: {exc}")
                self._nvml_handle = None

    def record_step(self, latency_s: float, samples: int = 1, tokens: int = 0):
        self.latencies.append(latency_s)
        self.samples += samples
        self.tokens += tokens
        self._snapshot_resources()

    def _snapshot_resources(self):
        if psutil:
            try:
                self.cpu_samples.append(psutil.cpu_percent(interval=None))
                self.ram_samples.append(psutil.virtual_memory().percent)
            except Exception:
                pass

        if torch and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                self.max_gpu_mem = max(self.max_gpu_mem, torch.cuda.max_memory_allocated())
            except Exception:
                pass

        if self._nvml_handle:
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle)
                self.gpu_power_samples.append(power_mw / 1000.0)
            except Exception:
                pass

    def summary(self) -> Dict[str, Any]:
        wall = time.time() - self.start_time
        avg_latency = float(np.mean(self.latencies)) if self.latencies else 0.0
        throughput = self.samples / wall if wall > 0 else 0.0
        token_throughput = self.tokens / wall if wall > 0 else 0.0

        def _avg(values: list):
            return float(np.mean(values)) if values else None

        summary = {
            "wall_time_s": wall,
            "avg_latency_s": avg_latency,
            "throughput_samples_per_s": throughput,
            "throughput_tokens_per_s": token_throughput,
            "cpu_percent_avg": _avg(self.cpu_samples),
            "ram_percent_avg": _avg(self.ram_samples),
            "gpu_power_w_avg": _avg(self.gpu_power_samples),
            "max_gpu_mem_gb": self.max_gpu_mem / (1024**3) if self.max_gpu_mem else None,
            "steps_recorded": len(self.latencies),
        }
        return summary

    def close(self):
        if self._nvml_handle:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
