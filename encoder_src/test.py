"Testing code for encoders"

import argparse

from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer
from utils.utils import (
    compute_metrics,
    load_rate_ner_dataset,
    tokenize_and_align_labels,
)

BATCH_SIZE = 16


def test_model(model_name: str, data_dir: str = "all"):
    print(f"Testing model: {model_name}")

    dataset, idx2label, label2idx = load_rate_ner_dataset(directory=data_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_path = f"./final_model_{model_name.replace('/', '_')}"
    model = AutoModelForTokenClassification.from_pretrained(model_path)

    # TODO ADD RESOURCE TRACKING HERE
    tokenized_test = dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label2idx), batched=True
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, idx2label),
    )

    # TODO ADD RESOURCE TRACKING HERE
    test_results = trainer.evaluate(tokenized_test["test"])
    print(f"Test micro F1 (strict exact span): {test_results['eval_micro_f1']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test NER model")
    parser.add_argument(
        "--model_name", type=str, required=True, help="Pretrained encoder name"
    )
    parser.add_argument(
        "--data_dir", type=str, default="all", help="Directory of RaTE-NER dataset"
    )
    args = parser.parse_args()
    test_model(args.model_name, args.data_dir)
