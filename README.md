# LLM Spam Classifier — GPT‑2 Spam Detection & Text Generation

A modular, production‑ready project for:

- Creating and balancing the SMS Spam dataset
- Building GPT‑2 from scratch
- Loading pretrained GPT‑2 weights
- Running text generation
- Preparing for future fine‑tuning and classifier training

The project is structured for clarity, extensibility, and real‑world engineering workflows.

---

## 📁 Project Structure

llm_spam_classifier/
│
├── run_everything_standalone.py     # End‑to‑end pipeline runner
│
├── data_prep/
│   ├── prepare_data.py              # Download, balance, split dataset
│
├── dataset/
│   ├── dataset.py                   # SpamDataset class
│
├── llm_config/
│   ├── config.py                    # GPT‑2 model configs
│
├── llm_model/
│   ├── transformerBlock.py
│   ├── layerNorm.py
│
├── llm_training/
│   ├── train_classifier.py          # Training loop (future)
│
├── llm_inference/
│   ├── generate.py                  # Text generation utilities
│
├── load_pre_trained_weight/
│   ├── gpt_download3.py             # GPT‑2 weight downloader
│   ├── load_weight.py               # Load pretrained GPT‑2 weights
│
└── data/                            # Generated dataset splits


---

## ⭐ Dataset Creation (High‑Level Summary)

### 1. Loads the SMS Spam Collection dataset
- Reads the extracted TSV file (`SMSSpamCollection.tsv`)
- Correct tab‑separated parsing
- Assigns proper column names: **Label**, **Text**

### 2. Balances the dataset
- Undersamples majority class (**ham**)
- Produces a perfectly balanced dataset: **747 ham / 747 spam**
- Ensures stable classifier training

### 3. Converts labels
- `"ham"` → **0**
- `"spam"` → **1**
- Matches the classifier’s 2‑class output head

### 4. Splits into train / validation / test
- Uses `random_split()`
- **70%** train
- **10%** validation
- **20%** test
- Ensures clean evaluation

### 5. Saves dataset splits
- Creates `data/` folder if missing
- Saves:
  - `data/train.csv`
  - `data/validation.csv`
  - `data/test.csv`

---

## ⭐ Dataset Loading & Dataloader Pipeline

### 6. GPT‑2 Tokenization
- Converts each message into token IDs
- No cleaning — raw text preserved
- Captures real spam patterns (`WINNER`, `$1000`, `FREE`, etc.)

### 7. Computes max sequence length
- Automatically finds longest message
- Prints `max_length`
- Ensures consistent padding

### 8. Pads all sequences
- Uses GPT‑2 pad token (**50256**)
- Produces uniform shape: `[batch_size, max_length]`
- Required for batching + GPU efficiency

### 9. Creates PyTorch dataloaders
- Train / validation / test loaders
- Correct batch sizes
- Shuffling enabled for training
- Yields tensors with correct shapes

---

## ⭐ GPT‑2 Backbone Loading

### 10. Loads pretrained GPT‑2 backbone
- Loads GPT‑2 small architecture
- Restores pretrained weights
- Embeddings, positional embeddings, transformer blocks restored
- Prints **"Backbone loaded successfully."**

### 11. Saves pretrained backbone
- Saves to `gpt2_backbone.pth`
- Ensures reproducibility
- Enables fast reloads
- Avoids repeated downloads

---

## ⭐ Text Generation

### 12. GPT‑2 Generation
- Tokenizes prompt
- Performs autoregressive generation
- Produces coherent GPT‑2 output
- Confirms backbone functionality before fine‑tuning

---

## 🚀 Running the Full Pipeline

### Run everything (data prep → model loading → generation)

From the parent directory:

```bash
python -m llm_spam_classifier.run_everything_standalone