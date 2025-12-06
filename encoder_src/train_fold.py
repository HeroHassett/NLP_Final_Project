"Training code for encoder models"

import argparse
import os

import numpy as np
from sklearn.model_selection import KFold
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from utils.utils import (
    compute_metrics,
    load_rate_ner_dataset,
    tokenize_and_align_labels,
)

from datasets import concatenate_datasets

# Default hyperparameters (will be tuned in grid)
NUM_EPOCHS = 3
BATCH_SIZE = 128
LEARNING_RATE = 2e-5
K_FOLDS = 5


def train_model(model_name: str, data_dir: str = "all"):
    print(f"Training with model: {model_name}")

    # Load dataset
    dataset, idx2label, label2idx = load_rate_ner_dataset(directory=data_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize entire dataset
    tokenized_dataset = dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label2idx), batched=True
    )

    # Merge train + validation for cross-validation
    full_train_dataset = concatenate_datasets([tokenized_dataset["train"], tokenized_dataset["validation"]])
    train_indices = list(range(len(full_train_dataset)))
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    # Hyperparameter grid
    HYPERPARAM_GRID = {
        "learning_rate": [2e-5, 3e-5],
        "per_device_train_batch_size": [64, 128],
        "num_train_epochs": [3, 5],
    }

    best_hyperparams = None
    best_micro_f1 = -1

    # Grid search
    for lr in HYPERPARAM_GRID["learning_rate"]:
        for batch_size in HYPERPARAM_GRID["per_device_train_batch_size"]:
            for epochs in HYPERPARAM_GRID["num_train_epochs"]:
                print(f"\n=== Hyperparams: LR={lr}, BATCH={batch_size}, EPOCHS={epochs} ===")
                fold_results = []

                for fold, (train_idx, val_idx) in enumerate(kfold.split(train_indices)):
                    print(f"\n----- Fold {fold + 1} -----")
                    train_split = full_train_dataset.select(train_idx)
                    val_split = full_train_dataset.select(val_idx)

                    model = AutoModelForTokenClassification.from_pretrained(
                        model_name, num_labels=len(label2idx)
                    )

                    os.environ["WANDB_PROJECT"] = "test-project"
                    os.environ["WANDB_LOG_MODEL"] = "checkpoint"
                    os.environ["WANDB_WATCH"] = "false"

                    training_args = TrainingArguments(
                        output_dir=f"./results_{model_name.replace('/', '_')}_lr{lr}_bs{batch_size}_ep{epochs}_fold{fold}",
                        eval_strategy="steps",
                        eval_steps=500,
                        logging_steps=200,
                        save_steps=1000,
                        learning_rate=lr,
                        per_device_train_batch_size=batch_size,
                        per_device_eval_batch_size=batch_size,
                        num_train_epochs=epochs,
                        weight_decay=0.01,
                        save_total_limit=2,
                        report_to="wandb",
                        bf16=True,
                    )

                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=train_split,
                        eval_dataset=val_split,
                        tokenizer=tokenizer,
                        compute_metrics=lambda p: compute_metrics(p, idx2label),
                    )

                    trainer.train()
                    eval_results = trainer.evaluate()
                    fold_results.append(eval_results)

                avg_micro_f1 = np.mean([r["eval_micro_f1"] for r in fold_results])
                print(f"Average micro F1 for these hyperparams: {avg_micro_f1:.4f}")

                if avg_micro_f1 > best_micro_f1:
                    best_micro_f1 = avg_micro_f1
                    best_hyperparams = {"learning_rate": lr, "batch_size": batch_size, "epochs": epochs}

    print(f"\nBest hyperparameters: {best_hyperparams} with micro F1={best_micro_f1:.4f}")

    # Retrain final model on full train+validation using best hyperparameters
    final_model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(label2idx)
    )

    final_args = TrainingArguments(
        output_dir=f"./final_model_{model_name.replace('/', '_')}",
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=200,
        save_steps=1000,
        learning_rate=best_hyperparams["learning_rate"],
        per_device_train_batch_size=best_hyperparams["batch_size"],
        per_device_eval_batch_size=best_hyperparams["batch_size"],
        num_train_epochs=best_hyperparams["epochs"],
        weight_decay=0.01,
        save_total_limit=2,
        report_to="wandb",
        bf16=True,
    )

    final_trainer = Trainer(
        model=final_model,
        args=final_args,
        train_dataset=full_train_dataset,
        eval_dataset=tokenized_dataset["test"],  # test set evaluation
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, idx2label),
    )

    final_trainer.train()
    final_trainer.save_model(f"./final_model_{model_name.replace('/', '_')}")
    print(f"Final model saved to ./final_model_{model_name.replace('/', '_')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NER model")
    parser.add_argument(
        "--model_name", type=str, required=True, help="Pretrained encoder name"
    )
    parser.add_argument(
        "--data_dir", type=str, default="all", help="Directory of RaTE-NER dataset"
    )
    args = parser.parse_args()
    train_model(args.model_name, args.data_dir)
