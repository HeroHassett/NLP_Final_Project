"Training code for encoder models"

import argparse

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

NUM_EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
K_FOLDS = 5


def train_model(model_name: str, data_dir: str = "all"):
    print(f"Training with model: {model_name}")

    # Load dataset
    dataset, idx2label, label2idx = load_rate_ner_dataset(directory=data_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize
    tokenized_dataset = dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label2idx), batched=True
    )

    train_dataset = tokenized_dataset["train"]
    fold_results = []

    from datasets import Dataset

    train_indices = list(range(len(train_dataset)))

    from sklearn.model_selection import KFold

    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_indices)):
        print(f"\n===== Fold {fold + 1} =====")
        # TODO NEED TO TRACK RESOURCE USAGE HERE
        train_split = train_dataset.select(train_idx)
        val_split = train_dataset.select(val_idx)

        model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=len(label2idx)
        )

        training_args = TrainingArguments(
            output_dir=f"./results_{model_name.replace('/', '_')}_fold{fold}",
            eval_strategy="steps",
            eval_steps=500,
            logging_steps=100,
            save_steps=1000,
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=NUM_EPOCHS,
            weight_decay=0.01,
            save_total_limit=2,
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
    print(f"\nAverage micro F1 across {K_FOLDS} folds: {avg_micro_f1:.4f}")

    final_model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(label2idx)
    )

    final_args = TrainingArguments(
        output_dir=f"./final_model_{model_name.replace('/', '_')}",
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=100,
        save_steps=1000,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        save_total_limit=2,
    )

    final_trainer = Trainer(
        model=final_model,
        args=final_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, idx2label),
    )

    # EITHER REFACTOR TO PYTORCH LOOP OR SOME KIND OF HOOK
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
