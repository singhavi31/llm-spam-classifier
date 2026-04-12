import pandas as pd
import torch
from torch.utils.data import Dataset

# Previously, we utilized a sliding window technique to generate uniformly
# sized text chunks, which were then grouped into batches for more efficient model training.
# Each chunk functioned as an individual training instance

# In the case of email spam classification, have two primary options:
# (1) Truncate all messages to the length of the shortest message in the
# dataset or batch.
# (2) Pad all messages to the length of the longest message in the dataset or
# batch.
# Option 1 is computationally cheaper, but it may result in significant information loss if
# shorter messages are much smaller than the average or longest messages, potentially
# reducing model performance. 

# So, we opt for the second option, which preserves the entire
# content of all messages.

# To implement option 2, where all messages are padded to the length of the longest
# message in the dataset, we add padding tokens to all shorter messages. 

# For this purpose, we use "<|endoftext|>" as a padding token, as discussed in chapter 2.
# However, instead of appending the string "<|endoftext|>" to each of the text messages
# directly, we can add the token ID corresponding to "<|endoftext|>" to the encoded text
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [tokenizer.encode(t) for t in self.data["Text"]]

        # Determine max_length
        if max_length is None:
            self.max_length = max(len(x) for x in self.encoded_texts)
            print(f"Max length determined: {csv_file} - {self.max_length}")
        else:
            self.max_length = max_length
            # Truncate sequences longer than max_length
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        # Pad all sequences to self.max_length
        self.encoded_texts = [
            x[:self.max_length] + [pad_token_id] * (self.max_length - len(x))
            for x in self.encoded_texts
        ]

    def __getitem__(self, idx):
        return (
            torch.tensor(self.encoded_texts[idx], dtype=torch.long),
            torch.tensor(self.data.iloc[idx]["Label"], dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)
