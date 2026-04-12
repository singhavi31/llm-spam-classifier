# LLM Project — GPT‑2 Spam Classification & Text Generation

This project provides a clean, modular, production‑ready structure for:

- Preparing the SMS Spam dataset  
- Building and loading a GPT‑2 model from scratch  
- Loading pretrained GPT‑2 weights  
- Running text generation  
- Preparing for future fine‑tuning and training  

Training logic is intentionally not included yet — the project is structured so it can be added later without breaking anything.

---

## 📁 Project Structure

llm_project/
│
├── all_run.py                     # Master script (run any part of the pipeline)
│
├── data_prep/
│   ├── prepare_data.py            # Download, balance, split dataset
│
├── model_setup/
│   ├── load_model.py              # Build GPT-2 model + load pretrained weights
│
├── training/
│   ├── run_training.py            # Placeholder for future training loop
│
├── inference/
│   ├── run_generation.py          # Text generation using GPT-2
│
├── llm_finetuning/
│   ├── preprocessing.py           # Balancing + splitting logic
│   ├── dataset.py                 # SpamDataset class
│   ├── config.py                  # GPT-2 configs
│   ├── model_loader.py            # GPTModel + weight loading
│   ├── generate.py                # Text generation utilities
│
├── llm_scratch/
│   ├── transformerBlock.py
│   ├── layerNorm.py
│
├── scripts/
│   ├── download_spam_dataset.py   # Download + unzip UCI SMS Spam dataset
│
└── gpt_download3.py               # GPT-2 weight downloader


### Run everything (data prep → model loading → generation)
<!-- This will:
Download the dataset
Prepare balanced train/val/test splits
Build datasets + dataloaders
Load GPT‑2 model + pretrained weights
Generate text -->

```bash
python -m llm_project.all_run


# Using all_run.py
# Inside all_run.py, you can toggle each step:
RUN_DATA_DOWNLOAD = True
RUN_DATA_PREP = True
RUN_DATALOADERS = True
RUN_MODEL_LOADING = True
RUN_GENERATION = True

# Example: Only run model loading + generation
RUN_DATA_DOWNLOAD = False
RUN_DATA_PREP = False
RUN_DATALOADERS = False
RUN_MODEL_LOADING = True
RUN_GENERATION = True

# Example: Only prepare data
RUN_DATA_DOWNLOAD = True
RUN_DATA_PREP = True
RUN_DATALOADERS = False
RUN_MODEL_LOADING = False
RUN_GENERATION = False

📦 What Each Module Does
data_prep/
Handles dataset preparation:
Download ZIP
Extract
Balance ham/spam
Split into train/val/test
Save CSVs
model_setup/
Builds GPT‑2 architecture and loads pretrained weights.
llm_finetuning/
Reusable components:
Dataset class
Preprocessing utilities
GPT‑2 configs
GPT‑2 model loader
Text generation utilities
inference/
Runs text generation using GPT‑2.
training/
Placeholder for your future training loop.

# below will load the model in memory and then we run only the text generation
#python -m llm_project.repl_generate

# python -m llm_project.run_all --generate --prompt "Write a motivational quote" --tokens 40 --temp 0.8