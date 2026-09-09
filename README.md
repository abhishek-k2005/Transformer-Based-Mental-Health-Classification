# ?? Transformer-Based Mental Health Classification

> A production-grade NLP system that classifies text (journal entries, social media posts, etc.) into **7 mental health categories** in real time — served through a premium dark-mode Streamlit web application.

**Built by:** Abhishek K · **Completed:** September 2026 · **Status:** ? Fully Deployed & Live

---

## ?? What It Does

Paste any piece of text — a journal entry, a social media post, or a paragraph — and the model will instantly predict which of the 7 mental health categories it most likely belongs to, along with a full probability distribution across all classes.

### ??? Classification Labels

| Label | Emoji | Confidence Color |
|---|---|---|
| Anxiety | ?? | Amber `#F59E0B` |
| Bipolar | ?? | Purple `#8B5CF6` |
| Depression | ?? | Indigo `#6366F1` |
| Normal | ?? | Green `#10B981` |
| Personality Disorder | ?? | Pink `#EC4899` |
| Stress | ?? | Orange `#F97316` |
| Suicidal | ?? | Red `#EF4444` |

---

## ??? Model Architecture

| Attribute | Detail |
|---|---|
| **Base Model** | `distilroberta-base` (DistilRoBERTa) |
| **Task Head** | `RobertaForSequenceClassification` |
| **Hidden Layers** | 6 Transformer encoder blocks |
| **Hidden Size** | 768 |
| **Attention Heads** | 12 per layer |
| **Vocab Size** | 50,265 tokens |
| **Max Token Length** | 192 (with truncation) |
| **Model Variant** | v5 (best across 5 training iterations) |
| **Model Weights** | ~313 MB (`model.safetensors`) |

---

## ?? Training Results

| Epoch | Accuracy | Macro F1 | Eval Loss |
|---|---|---|---|
| 1 | 74.87% | 70.10% | 0.6954 |
| **2 (Final)** | **77.10%** | **71.38%** | **0.6914** |

> ?? Best checkpoint: **Step 1500** · Macro F1 = **71.38%** · Accuracy = **77.1%**
> ? Inference speed: **~1 second** on CPU

---

## ?? Project Structure

```
Transformer-Based Mental Health Classification/
¦
+-- ?? Deep_Learning_Mental_Health_Sentiment_Analysis.ipynb  ? Full training notebook
+-- ?? app.py              ? Streamlit web application (UI + routing)
+-- ?? config.py           ? Central configuration (paths, labels, device)
+-- ?? inference.py        ? Model loading + prediction logic
+-- ?? preprocessing.py    ? Text cleaning pipeline (URLs, mentions, etc.)
+-- ?? requirements.txt    ? Dependency manifest
+-- ?? .gitignore
¦
+-- ?? data/
¦   +-- Combined Data.csv  ? Training corpus (~30 MB, ~7 balanced classes)
¦
+-- ?? models/
    +-- ?? final_model/    ? Active model artefacts
        +-- model.safetensors     ? Fine-tuned weights (313 MB)
        +-- tokenizer.json        ? Tokenizer vocabulary (3.4 MB)
        +-- config.json           ? Model configuration
        +-- trainer_state.json    ? Full training history & metrics
```

---

## ?? Tech Stack

| Layer | Technology |
|---|---|
| **Deep Learning** | PyTorch >= 2.0 |
| **NLP / Transformers** | ?? Hugging Face Transformers >= 4.40 |
| **Model Serialisation** | SafeTensors >= 0.4 |
| **Web App** | Streamlit >= 1.35 |
| **Inference Device** | CPU (portable; GPU-ready via `DEVICE` in `config.py`) |

---

## ?? Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/abhishek-k2005/Transformer-Based-Mental-Health-Classification.git
cd Transformer-Based-Mental-Health-Classification
```

### 2. Create & activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

> **Note:** The `models/final_model/` directory must be present with the downloaded weights (`model.safetensors`). If running from a fresh clone, download the model artefacts separately and place them in `models/final_model/`.

---

## ?? How It Works

```
User Input (raw text)
       ¦
       ?
preprocessing.py  ?  Lowercase · Strip URLs · Remove @mentions · Collapse whitespace
       ¦
       ?
inference.py      ?  DistilRoBERTa Tokenizer (max_length=192) ? PyTorch Tensors
       ¦
       ?
DistilRoBERTa     ?  6× Transformer Encoder ? CLS Pooling ? Linear(768 ? 7)
       ¦
       ?
Softmax           ?  7-class probability distribution
       ¦
       ?
app.py (Streamlit) ?  Prediction badge + confidence % + animated probability bars
```

---

## ?? App UI Highlights

- ?? Deep gradient dark background (`#0f0c29 ? #302b63 ? #24243e`)
- ? Glassmorphism result cards with `backdrop-filter: blur`
- ?? Gradient hero title (purple ? blue ? green)
- ??? Per-class color-coded prediction badges
- ?? Animated probability bars (smooth 0.6s ease transitions)
- ?? FadeIn slide-up animation on result reveal

---

## ?? Key Engineering Decisions

| Decision | Rationale |
|---|---|
| **DistilRoBERTa** over BERT | 40% fewer parameters, 60% faster, retains 97% of BERT performance |
| **MAX_LENGTH = 192** | Enough context for social/journal text; keeps inference fast |
| **`local_files_only=True`** | Fully offline — no internet needed at runtime |
| **`torch.no_grad()`** | Disables gradients during inference — saves memory & time |
| **`@st.cache_resource`** | Model loads once per session, not on every request |
| **SafeTensors format** | Safer, faster serialisation compared to `.bin` pickle format |
| **Preprocessing parity** | `clean_text()` in `preprocessing.py` exactly mirrors notebook training step |

---

## ?? Training Notebook

The Jupyter notebook [`Deep_Learning_Mental_Health_Sentiment_Analysis.ipynb`](./Deep_Learning_Mental_Health_Sentiment_Analysis.ipynb) covers:

- ?? Exploratory Data Analysis (class distribution, text length stats)
- ?? Preprocessing pipeline design
- ?? Tokenisation with DistilRoBERTa
- ?? Model experiments v1 ? v5 (hyperparameter tuning)
- ?? Evaluation metrics, confusion matrix, loss curves
- ?? Checkpoint saving & best model selection

---

## ?? Disclaimer

> This tool is intended for **research and portfolio purposes only**.
> It is **not** a clinical diagnostic instrument and should **not** be used as a substitute for professional mental health advice.
> If you or someone you know is struggling, please seek help from a qualified mental health professional.

---

## ?? License

This project is open-source and available under the [MIT License](LICENSE).

---

*Built with ?? by [Abhishek K](https://github.com/abhishek-k2005) — demonstrating the full ML engineering lifecycle from raw data to a deployed transformer-based NLP application.*
