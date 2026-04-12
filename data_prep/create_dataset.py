import pandas as pd

def create_balanced_dataset(df):
    # Count the instances of "spam" (Count how many spam messages exist)
    # If you have 747 spam messages, num_spam = 747.
    num_spam = df[df["Label"] == "spam"].shape[0]

    # Randomly sample "ham" instances to match the number of "spam" instances (Randomly sample the same number of ham messages)
    # If you originally had 4,000 ham messages, this line randomly picks 747 ham messages to match the spam count.
    # This prevents the classifier from being biased toward the majority class.
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)

    # Combine ham "subset" with "spam"
    return pd.concat([ham_subset, df[df["Label"] == "spam"]])

def random_split(df, train_frac, validation_frac):
    # We create a random_split function to split the dataset into three parts: 70% for
    # training, 10% for validation, and 20% for testing. 
    # (These ratios are common in machine
    # learning to train, adjust, and evaluate models.)   
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    val_end = train_end + int(len(df) * validation_frac)
    return df[:train_end], df[train_end:val_end], df[val_end:]
