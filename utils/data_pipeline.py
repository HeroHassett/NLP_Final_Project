from datasets import load_dataset, ClassLabel, DatasetDict, Sequence
import requests

def load_rate_ner_dataset(directory="all", ner_type="IOB"):
    """
    Loads only the [all] IOB splits, downloads idx2label.json and label2idx.json,
    and converts ner_tags to Sequence(ClassLabel).
    
    Returns:
        dataset (DatasetDict): Hugging Face dataset with train/validation/test splits
        idx2label (dict): mapping from integer IDs to label strings
        label2idx (dict): mapping from label strings to integer IDs
    """
    # 1. Download label mapping JSONs from the dataset hub
    idx2label_url = "https://huggingface.co/datasets/Angelakeke/RaTE-NER/resolve/main/idx2label.json"
    label2idx_url = "https://huggingface.co/datasets/Angelakeke/RaTE-NER/resolve/main/label2idx.json"

    try:
        idx2label = requests.get(idx2label_url).json()
        label2idx = requests.get(label2idx_url).json()
        label_names = [idx2label[str(i)] for i in range(len(idx2label))]
        ner_classlabel = ClassLabel(names=label_names)
        ner_sequence = Sequence(feature=ner_classlabel)
    except Exception as e:
        print(f"Error in label2idx or idx2label: {e}")
        label2idx = {}
        idx2label = {}
    
    data_files = {
        "train": f"{directory}/train_{ner_type}.json",
        "validation": f"{directory}/dev_{ner_type}.json",
        "test": f"{directory}/test_{ner_type}.json"
    }

    dataset = load_dataset("Angelakeke/RaTE-NER", data_files=data_files)


    # 4. Convert ner_tags to Sequence(ClassLabel)
    if label2idx and idx2label:
        for split in ["train", "validation", "test"]:
            dataset[split] = dataset[split].cast_column("ner_tags", ner_sequence)
    return dataset, idx2label, label2idx


# Example usage if run directly
if __name__ == "__main__":
    dataset, idx2label, label2idx = load_rate_ner_dataset()
    print("Dataset splits:", dataset)
    print("Example from train split:", dataset["train"][0])
    print("idx2label:", idx2label)
    print("label2idx:", label2idx)
    print("label2idx:", label2idx)
