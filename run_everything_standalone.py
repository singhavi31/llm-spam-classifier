# run_everything_standalone.py
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
import tiktoken

# from .inference.run_generation import run_generation
from .dataset.dataset import SpamDataset
from .inference.inference import classify_review
from .load_pre_trained_weight.load_weight import load_gpt2_backbone
from .finetuning.model_finetune import prepare_model_for_classification
from .data_prep.create_dataset import create_balanced_dataset, random_split
from .training.train_classifier import train_classifier_simple
from .data_prep.download_spam_dataset import download_and_unzip_spam_data



def run_data_prep():
    print("\n=== DATA PREPARATION ===")
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "sms_spam_collection.zip"
    extracted_path = "sms_spam_collection"
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

    download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    balanced_df = create_balanced_dataset(df)
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
    print('balanced_df["Label"].value_counts()', balanced_df["Label"].value_counts())

    train_df, val_df, test_df = random_split(balanced_df, 0.7, 0.1)

    Path("data").mkdir(exist_ok=True)
    train_df.to_csv("data/train.csv", index=False)
    val_df.to_csv("data/validation.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    print("Data preparation complete.")

def decode_batch(input_batch, tokenizer, pad_token_id=50256):
    decoded = []
    for seq in input_batch:
        seq = seq.tolist()
        seq = [t for t in seq if t != pad_token_id]
        decoded.append(tokenizer.decode(seq))
    return decoded

def run_dataset_loading():
    print("\n=== LOADING DATASETS ===")

    tokenizer = tiktoken.get_encoding("gpt2")

    train_dataset = SpamDataset("data/train.csv", tokenizer)
    val_dataset = SpamDataset(
        "data/validation.csv", tokenizer, max_length=train_dataset.max_length
    )
    test_dataset = SpamDataset(
        "data/test.csv", tokenizer, max_length=train_dataset.max_length
    )

    num_workers = 0
    batch_size = 8
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False
    )

    print("Datasets loaded successfully.")
    return train_loader, val_loader, test_loader, tokenizer

def run_model_build():
    print("\n=== BUILDING GPT-2 SMALL BACKBONE WITH PRETRAINED WEIGHTS ===")
    model = load_gpt2_backbone(model_name="gpt2-small (124M)", models_dir="gpt2")
    print("GPT-2 pretrained backbone loaded.")
    return model

def run_model_finetuning(model):
    print("\n=== APPLYING CLASSIFICATION FINETUNING ===")
    model = prepare_model_for_classification(model, num_classes=2)
    return model

def run_training(model, train_loader, val_loader):
    print("\n=== TRAINING CLASSIFIER ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

    num_epochs = 5
    train_losses, val_losses, train_accs, val_accs, examples_seen = (
        train_classifier_simple(
            model,
            train_loader,
            val_loader,
            optimizer,
            device,
            num_epochs=num_epochs,
            eval_freq=50,
            eval_iter=5,
        )
    )

    print("\nTraining complete.")

    save_path = "gpt2_spam_classifier.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Trained model saved to {save_path}")

    return model

def run_prediction(model, tokenizer, text):
    print("\n=== SPAM CLASSIFICATION ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    result = classify_review(text, model, tokenizer, device)
    print(f"\nInput: {text}")
    print(f"Prediction: {result}")

# ============================================================
# CHAT MODE
# ============================================================

def chat_mode(model, tokenizer):
    print("\n=== CHAT MODE: SPAM CLASSIFIER ===")
    print("Type a message to classify it. Type 'quit' to exit.\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Exiting chat mode.")
            break

        prediction = classify_review(user_input, model, tokenizer, device)
        print(f"Model: {prediction}\n")
# ============================================================
# MAIN: RUN EVERYTHING IN ONE GO
# ============================================================


def main():
    run_data_prep()
    train_loader, val_loader, test_loader, tokenizer = run_dataset_loading()
    model = run_model_build()
    model = run_model_finetuning(model)
    model = run_training(model, train_loader, val_loader)

    # Optional: quick sanity generation
    # run_generation(model, tokenizer)

    # Optional: quick prediction example
    sample_text = (
        "Congratulations! You have won a free ticket. Call now to claim your prize."
    )
    run_prediction(model, tokenizer, sample_text)

    messages = [
        "Hi, the SEXYCHAT girls are waiting for you to text them",
        "Let's meet tomorrow.",
        "URGENT: Your account is locked.",
        "You are a winner you have been specially selected to receive $1000 cash or a $2000 award.",
    ]

    for msg in messages:
        run_prediction(model, tokenizer, msg)

    # Start interactive chat mode
    chat_mode(model, tokenizer)


if __name__ == "__main__":
    main()
