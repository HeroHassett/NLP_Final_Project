# Prompted decoder inference for RaTE-NER
import argparse
import json
import os
import time
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm.auto import tqdm


# Ensure repo root is on sys.path for `utils` imports when run as a script
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.decoder_utils import (
    ResourceTracker,
    compute_tag_metrics,
    load_rate_ner_dataset,
)
import certifi

os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['HF_HUB_CACHE'] = '/gpfs/scratch/jn2691/NLP_RESULTS'
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
    """
    Optimized PromptDecoder with:
      - batched generate support (preserves exact prompt per sample)
      - optional torch.compile
      - optional xformers memory-efficient attention (if available)
      - KV-cache warmup
    """

    def __init__(
        self,
        model_name: str,
        allowed_labels: Sequence[str],
        max_new_tokens: int = 48,
        temperature: float = 0.0,
        top_p: float = 1.0,
        device: str | None = None,
        hf_token: str | None = None,
        torch_compile: bool = False,
        use_xformers: bool = False,
        warmup_steps: int = 0,
    ):
        self.allowed_labels = list(allowed_labels)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.hf_token = hf_token or os.getenv("HUGGING_FACE_HUB_TOKEN")
        self.torch_compile = torch_compile
        self.use_xformers = use_xformers

        if ("llama" in model_name.lower()) and not self.hf_token:
            raise RuntimeError(
                "Hugging Face token required for gated Llama checkpoints. "
                "Pass --hf_token or set HUGGING_FACE_HUB_TOKEN."
            )

        # Use fast tokenizer if available
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=self.hf_token,
            use_fast=True,
            cache_dir="/gpfs/scratch/jn2691/NLP_RESULTS"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Keep padding on left so generated tokens always are appended at seq end
        self.tokenizer.padding_side = "left"

        # Load model
#        self.model = AutoModelForCausalLM.from_pretrained(
#            model_name,
#            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
#            device_map=None,
#            token=self.hf_token,
#            cache_dir="/gpfs/scratch/jn2691/NLP_RESULTS"
#
#        )
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map=None,
            token=self.hf_token,
            cache_dir="/gpfs/scratch/jn2691/NLP_RESULTS"

        )


        if self.device == "cuda":
            self.model = self.model.to(self.device)

        self.model.eval()

        # Optional: enable xformers / mem-efficient attention if available
        if self.use_xformers:
            try:
                # some HF model objects expose this helper
                if hasattr(self.model, "enable_xformers_memory_efficient_attention"):
                    self.model.enable_xformers_memory_efficient_attention()
                else:
                    # try a generic import (will fail if xformers not installed)
                    import xformers  # type: ignore
                    # nothing else to do; model will use xformers if supported
                print("xFormers/mem-efficient attention enabled (if supported).")
            except Exception as e:
                print("Warning: could not enable xformers/mem-efficient attention:", e)

        # Optional: torch.compile speedups (PyTorch 2.0+)
        if self.torch_compile and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)
                print("Model compiled with torch.compile().")
            except Exception as e:
                print("Warning: torch.compile() failed or not beneficial:", e)

        # Warm up KV caches / CUDA kernels (optional)
        if warmup_steps and self.device == "cuda":
            try:
                self._warmup(warmup_steps)
            except Exception as e:
                print("Warmup failed (continuing):", e)

    def _warmup(self, steps: int = 2) -> None:
        """
        Run a few small generates to warm up CUDA kernels and initialize caches.
        Keeps prompt identical by using a short, valid prompt.
        """
        dummy_prompt = "Hello\n\nJSON:"
        enc = self.tokenizer(dummy_prompt, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            for _ in range(steps):
                # small generation to populate kernels & caches
                _ = self.model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc.get("attention_mask"),
                    max_new_tokens=1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

    def generate_tags(self, tokens: Sequence[str]) -> Dict[str, Any]:
        """
        Single-sample wrapper that reuses the batched tokenizer path.
        """
        out = self.generate_tags_batch([tokens])
        return out[0]

    def generate_tags_batch(self, batch_tokens: List[Sequence[str]]) -> List[Dict[str, Any]]:
        """
        Batch multiple samples (preserves each sample's exact prompt).
        Returns list of outputs with same structure as single generate_tags.
        """

        # Build prompts for each sample (unchanged prompt text)
        prompts = [_format_prompt(tokens, self.allowed_labels) for tokens in batch_tokens]

        # Tokenize as a batch; pad on left so sequence ends align and generation appends at same side
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )

        input_ids = encoded["input_ids"].to(self.model.device)  # shape (B, S)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)

        # compute each prompt length (number of non-pad tokens) to slice generated tokens per sample
        # attention_mask sums to actual token count per example
        input_lengths = attention_mask.sum(dim=1).long() if attention_mask is not None else torch.tensor([input_ids.shape[1]] * input_ids.shape[0], device=input_ids.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        results: List[Dict[str, Any]] = []
        # output_ids shape: (B, S_out)
        # For each batch item, slice from input_lengths[i] : S_out to get generated tokens
        for i in range(output_ids.shape[0]):
            in_len = int(input_lengths[i].item())
            gen_ids = output_ids[i][in_len:]
            decoded = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            tags = _parse_tag_response(decoded, len(batch_tokens[i]), self.allowed_labels)
            results.append(
                {
                    "tags": tags,
                    "raw_generation": decoded,
                    "prompt_tokens": in_len,
                    "generated_tokens": gen_ids.shape[0],
                }
            )

        return results


def _prepare_references(idx2label: Dict[str, str], ner_tags: Sequence[int]) -> List[str]:
    return [idx2label[str(i)] for i in ner_tags]


def _init_wandb(args: argparse.Namespace, label_set: Sequence[str]):
    if not wandb or not args.wandb_project:
        return None
    run = wandb.init(
        project=args.wandb_project,
        dir='/gpfs/scratch/jn2691/NLP_Results',
        name=args.wandb_run_name,
        config={
            "model_name": args.model_name,
            "split": args.split,
            "max_new_tokens": args.max_new_tokens,
            "top_p": args.top_p,
            "label_set": list(label_set),
            "max_samples": args.max_samples,
            "batch_size": args.batch_size,
            "torch_compile": args.torch_compile,
            "use_xformers": args.use_xformers,
            "warmup_steps": args.warmup_steps,
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
        torch_compile=args.torch_compile,
        use_xformers=args.use_xformers,
        warmup_steps=args.warmup_steps,
    )

    tracker = ResourceTracker()
    wb_run = _init_wandb(args, allowed_labels)

    predictions: List[List[str]] = []
    references: List[List[str]] = []
    generations: List[Dict[str, Any]] = []
    prompt_token_total = 0
    gen_token_total = 0

    batch_size = max(1, args.batch_size)
    # We'll iterate through the dataset in batches (preserves sample order)
    examples = list(split_data)
    total_samples = len(examples)
    it = range(0, total_samples, batch_size)

    for batch_start in tqdm(it, desc="Decoding", total=(total_samples + batch_size - 1) // batch_size):
        batch_examples = examples[batch_start : batch_start + batch_size]
        tokens_batch = [ex["tokens"] for ex in batch_examples]
        gold_labels_batch = [_prepare_references(idx2label, ex["ner_tags"]) for ex in batch_examples]

        start = time.time()
        outputs = decoder.generate_tags_batch(tokens_batch)
        latency = time.time() - start

        # outputs aligns with tokens_batch order
        for i, out in enumerate(outputs):
            pred_tags = out["tags"]
            gold_labels = gold_labels_batch[i]

            # Ensure length match
            if len(pred_tags) < len(gold_labels):
                pred_tags += ["O"] * (len(gold_labels) - len(pred_tags))
            elif len(pred_tags) > len(gold_labels):
                pred_tags = pred_tags[: len(gold_labels)]

            predictions.append(pred_tags)
            references.append(gold_labels)
            prompt_token_total += out.get("prompt_tokens", 0)
            gen_token_total += out.get("generated_tokens", 0)
            generations.append(
                {
                    "tokens": tokens_batch[i],
                    "gold": gold_labels,
                    "pred": pred_tags,
                    "raw_generation": out["raw_generation"],
                }
            )

        tracker.record_step(
            latency_s=latency,
            samples=len(batch_examples),
            tokens=sum(len(ex["tokens"]) for ex in batch_examples) + sum(out.get("generated_tokens", 0) for out in outputs),
        )

        if wb_run:
            # log aggregated stats for this batch (step = number of processed samples so far)
            step = len(predictions)
            wandb.log(
                {
                    "batch_start": batch_start,
                    "batch_size": len(batch_examples),
                    "latency_s": latency,
                    "prompt_tokens": sum(out.get("prompt_tokens", 0) for out in outputs),
                    "generated_tokens": sum(out.get("generated_tokens", 0) for out in outputs),
                    "processed_samples": step,
                },
                step=step,
            )

    metrics = compute_tag_metrics(predictions, references)
    resource_stats = tracker.summary()
    tracker.close()

    sample_count = len(predictions)
    avg_prompt_tokens = prompt_token_total / sample_count if sample_count else 0
    avg_generated_tokens = gen_token_total / sample_count if sample_count else 0

    print(f"Split: {args.split} | Samples: {len(predictions)}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Micro F1: {metrics['micro_f1']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Resource stats: {resource_stats}")
    print(f"Avg prompt tokens: {avg_prompt_tokens:.1f} | Avg generated tokens: {avg_generated_tokens:.1f}")

    if wb_run:
        wandb.log(
            {
                **metrics,
                **resource_stats,
                "num_samples": sample_count,
                "avg_prompt_tokens": avg_prompt_tokens,
                "avg_generated_tokens": avg_generated_tokens,
                "total_generated_tokens": gen_token_total,
            }
        )
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
    parser.add_argument("--max_new_tokens", type=int, default=48, help="Decoder max new tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 for greedy)")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling parameter")
    parser.add_argument("--device", type=str, default=None, help="Force device (e.g., cuda, cpu). Defaults to auto")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of samples to decode per generate() call (batching)")
    parser.add_argument("--torch_compile", action="store_true", help="Enable torch.compile() if available (may speed up)")
    parser.add_argument("--use_xformers", action="store_true", help="Try to enable xFormers / memory-efficient attention (if installed)")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of tiny generates to warm up CUDA kernels")
    parser.add_argument("--save_predictions", type=str, default="decoder_outputs/predictions.jsonl", help="Path to save raw generations")
    parser.add_argument("--wandb_project", type=str, default=DEFAULT_WANDB_PROJECT, help="W&B project name for logging")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Optional W&B run name")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token (else read from HUGGING_FACE_HUB_TOKEN)")
    return parser.parse_args(args=arg_list)


if __name__ == "__main__":
    args = parse_args()
    run_decoder(args)

