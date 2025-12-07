"W&B sweep launcher for decoder hyperparameters"

import argparse
from types import SimpleNamespace
from typing import Sequence

import wandb

from decoder_src.decode import (
    DEFAULT_DECODER_MODEL,
    DEFAULT_WANDB_PROJECT,
    parse_args as decode_parse_args,
    run_decoder,
)


def _base_args(overrides: dict) -> SimpleNamespace:
    base = decode_parse_args([])  # default args without CLI parsing side effects
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def build_sweep_config(
    temperature_values: Sequence[float],
    top_p_values: Sequence[float],
    max_new_tokens_values: Sequence[int],
    sweep_name: str,
) -> dict:
    return {
        "name": sweep_name,
        "method": "random",
        "metric": {"name": "micro_f1", "goal": "maximize"},
        "parameters": {
            "temperature": {"values": list(temperature_values)},
            "top_p": {"values": list(top_p_values)},
            "max_new_tokens": {"values": list(max_new_tokens_values)},
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Launch a W&B sweep for decoder hyperparameters.")
    parser.add_argument("--project", type=str, default=DEFAULT_WANDB_PROJECT, help="W&B project")
    parser.add_argument("--entity", type=str, default=None, help="W&B entity/org (optional)")
    parser.add_argument("--sweep_name", type=str, default="decoder-hparam-sweep", help="Sweep name")
    parser.add_argument("--count", type=int, default=10, help="Number of runs for the agent")
    parser.add_argument("--temperature_values", nargs="+", type=float, default=[0.0, 0.2, 0.4])
    parser.add_argument("--top_p_values", nargs="+", type=float, default=[0.85, 0.9, 1.0])
    parser.add_argument("--max_new_tokens_values", nargs="+", type=int, default=[96, 128, 160])
    parser.add_argument("--model_name", type=str, default=DEFAULT_DECODER_MODEL)
    parser.add_argument("--data_dir", type=str, default="all")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap per run (useful for quick sweeps)")
    args = parser.parse_args()

    sweep_config = build_sweep_config(
        temperature_values=args.temperature_values,
        top_p_values=args.top_p_values,
        max_new_tokens_values=args.max_new_tokens_values,
        sweep_name=args.sweep_name,
    )

    sweep_id = wandb.sweep(sweep=sweep_config, project=args.project, entity=args.entity)
    print(f"Created sweep: {sweep_id}")

    def _run():
        run = wandb.init(project=args.project, entity=args.entity)
        cfg = run.config

        decoded_args = _base_args(
            {
                "model_name": args.model_name,
                "data_dir": args.data_dir,
                "split": args.split,
                "device": args.device,
                "hf_token": args.hf_token,
                "wandb_project": args.project,
                "wandb_run_name": f"{args.sweep_name}-{run.id}",
                "max_samples": args.max_samples,
                "temperature": cfg.get("temperature", 0.0),
                "top_p": cfg.get("top_p", 1.0),
                "max_new_tokens": cfg.get("max_new_tokens", 128),
            }
        )
        run_decoder(decoded_args)

    wandb.agent(sweep_id, function=_run, count=args.count, project=args.project, entity=args.entity)


if __name__ == "__main__":
    main()
