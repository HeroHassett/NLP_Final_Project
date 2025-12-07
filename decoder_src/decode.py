"Prompted decoder inference for RaTE-NER"

import argparse
import json
import os
import time
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure repo root is on sys.path for `utils` imports when run as a script
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.utils import (
    ResourceTracker,
    compute_tag_metrics,
    load_rate_ner_dataset,
)

try:
    import wandb
except Exception:  # pragma: no cover - optional dependency at import time
    wandb = None

# Default checkpoint (open Llama 3.1 Instruct); Llama 4 stays available if you have access
DEFAULT_DECODER_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA4_DECODER_MODEL = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
DEFAULT_WANDB_PROJECT = "NLP-Final-Project"

# Hugging Face token is read from CLI arg or env (HUGGING_FACE_HUB_TOKEN); not hard-coded.

def _format_prompt(tokens: Sequence[str], allowed_labels: Sequence[str]) -> str:
    token_block = "\n".join(f"{i+1}. {tok}" for i, tok in enumerate(tokens))
    label_list = ", ".join(allowed_labels)
    return (
        "You are an expert clinical named entity recognizer for radiology reports.\n"
        "Tag each token with an IOB2 label from the allowed label set. "
        "Return JSON with a single key \"tags\" whose value is an array of strings.\n"
        f"Allowed labels (use exactly these strings): {label_list}.\n"
        f"The array must contain exactly {len(tokens)} items and align to the token order below.\n"
        "Example response: {\"tags\": [\"O\", \"B-FINDING\", \"I-FINDING\"]}\n\n"
        "Tokens:\n"
        f"{token_block}\n\n"
        "JSON:"
    )


def _parse_tag_response(
    raw_text: str, num_tokens: int, allowed_labels: Sequence[str]
) -> List[str]:
    """
    Attempt to parse a JSON object containing {"tags": [...]}.
    Falls back to trimming/padding with 'O' on schema or length errors.
    """
    cleaned = raw_text.strip()
    json_start = cleaned.find("{")
    if json_start != -1:
        cleaned = cleaned[json_start:]

    tags: List[str] = []
    try:
        parsed = json.loads(cleaned)
        tags = parsed.get("tags", [])
    except Exception:
        # Fallback: try to extract between first '[' and ']'
        l_idx, r_idx = cleaned.find("["), cleaned.find("]")
        if l_idx != -1 and r_idx != -1 and r_idx > l_idx:
            try:
                tags = json.loads(cleaned[l_idx : r_idx + 1])
            except Exception:
                tags = []

    # Normalize tags and enforce allowed label set
    normalized = []
    for tag in tags:
        if isinstance(tag, str) and tag in allowed_labels:
            normalized.append(tag)
        else:
            normalized.append("O")

    if len(normalized) < num_tokens:
        normalized.extend(["O"] * (num_tokens - len(normalized)))
    elif len(normalized) > num_tokens:
        normalized = normalized[:num_tokens]

    if not normalized:
        normalized = ["O"] * num_tokens
    return normalized


class PromptDecoder:
    def __init__(
        self,
        model_name: str,
        allowed_labels: Sequence[str],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        device: str | None = None,
        hf_token: str | None = None,
    ):
        self.allowed_labels = list(allowed_labels)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.hf_token = hf_token or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if ("llama" in model_name.lower()) and not self.hf_token:
            raise RuntimeError(
                "Hugging Face token required for gated Llama checkpoints. "
                "Pass --hf_token or set HUGGING_FACE_HUB_TOKEN."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=self.hf_token,
            use_auth_token=self.hf_token,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.padding_side != "left":
            self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16 if torch.cuda.is_available() else None,
            device_map="auto" if self.device == "cuda" else None,
            token=self.hf_token,
            use_auth_token=self.hf_token,
        )
        if self.device != "cuda":
            self.model.to(self.device)
        self.model.eval()

    def generate_tags(self, tokens: Sequence[str]) -> Dict[str, Any]:
        prompt = _format_prompt(tokens, self.allowed_labels)
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.model.device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][input_ids.shape[1] :]
        decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        tags = _parse_tag_response(decoded, len(tokens), self.allowed_labels)
        return {
            "tags": tags,
            "raw_generation": decoded,
            "prompt_tokens": input_ids.shape[1],
            "generated_tokens": generated_ids.shape[0],
        }


def _prepare_references(idx2label: Dict[str, str], ner_tags: Sequence[int]) -> List[str]:
    return [idx2label[str(i)] for i in ner_tags]


def _init_wandb(args: argparse.Namespace, label_set: Sequence[str]):
    if not wandb or not args.wandb_project:
        return None
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "model_name": args.model_name,
            "split": args.split,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "label_set": list(label_set),
            "max_samples": args.max_samples,
        },
    )
    return run


def run_decoder(args: argparse.Namespace):
    dataset, idx2label, _ = load_rate_ner_dataset(directory=args.data_dir, ner_type=args.ner_type)
    allowed_labels = [idx2label[str(i)] for i in range(len(idx2label))]

    split_data = dataset[args.split]
    if args.max_samples:
        split_data = split_data.select(range(min(args.max_samples, len(split_data))))

    decoder = PromptDecoder(
        model_name=args.model_name,
        allowed_labels=allowed_labels,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
        hf_token=args.hf_token,
    )

    tracker = ResourceTracker()
    wb_run = _init_wandb(args, allowed_labels)

    predictions: List[List[str]] = []
    references: List[List[str]] = []
    generations: List[Dict[str, Any]] = []

    for example in split_data:
        tokens = example["tokens"]
        gold_labels = _prepare_references(idx2label, example["ner_tags"])

        start = time.time()
        output = decoder.generate_tags(tokens)
        latency = time.time() - start

        pred_tags = output["tags"]
        # Ensure length match
        if len(pred_tags) < len(gold_labels):
            pred_tags += ["O"] * (len(gold_labels) - len(pred_tags))
        elif len(pred_tags) > len(gold_labels):
            pred_tags = pred_tags[: len(gold_labels)]

        predictions.append(pred_tags)
        references.append(gold_labels)
        generations.append(
            {
                "tokens": tokens,
                "gold": gold_labels,
                "pred": pred_tags,
                "raw_generation": output["raw_generation"],
            }
        )

        tracker.record_step(
            latency_s=latency,
            samples=1,
            tokens=len(tokens) + output.get("generated_tokens", 0),
        )

    metrics = compute_tag_metrics(predictions, references)
    resource_stats = tracker.summary()
    tracker.close()

    print(f"Split: {args.split} | Samples: {len(predictions)}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Micro F1: {metrics['micro_f1']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Resource stats: {resource_stats}")

    if wb_run:
        wandb.log({**metrics, **resource_stats, "num_samples": len(predictions)})
        wb_run.finish()

    if args.save_predictions:
        save_dir = os.path.dirname(args.save_predictions)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        with open(args.save_predictions, "w", encoding="utf-8") as f:
            for row in generations:
                f.write(json.dumps(row) + "\n")
        print(f"Saved generations to {args.save_predictions}")


def parse_args(arg_list: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prompted decoder inference for RaTE-NER")
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_DECODER_MODEL,
        help=f"Decoder model name (default: {DEFAULT_DECODER_MODEL})",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"], help="Dataset split to evaluate")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="all",
        help="Directory of RaTE-NER dataset shards. If files are absent, falls back to hosted dataset.",
    )
    parser.add_argument("--ner_type", type=str, default="IOB", help="NER tag type (matches dataset files)")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples for quick runs")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Decoder max new tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 for greedy)")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling parameter")
    parser.add_argument("--device", type=str, default=None, help="Force device (e.g., cuda, cpu). Defaults to auto")
    parser.add_argument("--save_predictions", type=str, default="decoder_outputs/predictions.jsonl", help="Path to save raw generations")
    parser.add_argument("--wandb_project", type=str, default=DEFAULT_WANDB_PROJECT, help="W&B project name for logging")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Optional W&B run name")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token (else read from HUGGING_FACE_HUB_TOKEN)")
    return parser.parse_args(args=arg_list)


if __name__ == "__main__":
    args = parse_args()
    run_decoder(args)
