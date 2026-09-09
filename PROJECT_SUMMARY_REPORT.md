# 🧠 Transformer-Based Mental Health Classification
### Final Project Summary Report

> **Built by:** Abhishek K · **Completed:** September 2026  
> **Status:** ✅ Fully Deployed & Live

---

## 🎉 What Was Built

A production-grade **NLP deep-learning system** that reads a piece of text — a journal entry, a social-media post, a paragraph — and classifies it into one of **7 mental health categories** in real time, served via a beautiful dark-mode web application.

---

## 🗂️ Project At a Glance

| Attribute | Detail |
|---|---|
| **Task** | Multi-class text classification (7 classes) |
| **Model Architecture** | DistilRoBERTa (fine-tuned) — `RobertaForSequenceClassification` |
| **Model Variant** | v5 (best checkpoint across 5 iterations) |
| **Dataset** | Combined Mental Health Dataset — `Combined Data.csv` (~30 MB) |
| **Max Token Length** | 192 tokens (truncation enabled) |
| **Training Epochs** | 2 epochs (1 500 steps) |
| **Inference Device** | CPU (fully portable; GPU-ready) |
| **Web Framework** | Streamlit |
| **Model Size (weights)** | ~313 MB (`model.safetensors`) |

---

## 🏷️ Classification Labels

| ID | Label | Color |
|---|---|---|
| 0 | 😰 Anxiety | Amber `#F59E0B` |
| 1 | 🟣 Bipolar | Purple `#8B5CF6` |
| 2 | 💙 Depression | Indigo `#6366F1` |
| 3 | 🟢 Normal | Green `#10B981` |
| 4 | 🩷 Personality Disorder | Pink `#EC4899` |
| 5 | 🟠 Stress | Orange `#F97316` |
| 6 | 🔴 Suicidal | Red `#EF4444` |

---

## 📊 Training Metrics

| Epoch | Accuracy | Macro F1 | Eval Loss | Training Loss |
|---|---|---|---|---|
| 1.0 | 74.87% | 70.10% | 0.6954 | ~1.40 |
| **2.0 (Final)** | **77.10%** | **71.38%** | **0.6914** | ~1.02 |

> 🏆 **Best checkpoint**: Step 1500 · Macro F1 = **0.7138** · Accuracy = **77.1%**

### Training Loss Curve

```
Loss
2.81 ┤●
2.00 ┤
1.80 ┤  ●
1.40 ┤     ●
1.27 ┤        ●
1.08 ┤           ●  ●  ●
     └─────────────────────── Steps
     200  400  600  800  1000 1400
```

The loss curve shows **consistent, healthy convergence** with no signs of overfitting — the model generalised well across both epochs.

---

## 🏗️ Architecture Deep-Dive

### DistilRoBERTa Configuration

| Parameter | Value |
|---|---|
| `model_type` | `roberta` |
| `num_hidden_layers` | **6** (distilled from 12) |
| `hidden_size` | **768** |
| `num_attention_heads` | **12** |
| `intermediate_size` | **3072** |
| `vocab_size` | **50,265** |
| `max_position_embeddings` | 514 |
| `hidden_act` | GELU |
| `attention_dropout` | 0.1 |
| `hidden_dropout` | 0.1 |

---

## 🔄 End-to-End Data Flow

```mermaid
flowchart TD
    A["🧑‍💻 User Input\n(Raw Text)"] --> B

    subgraph PREPROCESS ["⚙️ Preprocessing (preprocessing.py)"]
        B["Lowercase + Strip"] --> C["Remove URLs\nhttps:// / www."]
        C --> D["Remove @Mentions"]
        D --> E["Collapse Whitespace"]
    end

    E --> F

    subgraph TOKENIZE ["🔤 Tokenisation (inference.py)"]
        F["DistilRoBERTa Tokenizer\nmax_length=192, truncation=True"] --> G["Token IDs + Attention Mask\n→ PyTorch Tensor"]
    end

    G --> H

    subgraph MODEL ["🧠 DistilRoBERTa Model (models/final_model/)"]
        H["Embedding Layer\n(vocab_size=50265, hidden=768)"] --> I["6× Transformer Encoder Blocks\n(12 attention heads each)"]
        I --> J["Pooled CLS Representation\n(768-dim)"]
        J --> K["Classification Head\nLinear(768 → 7)"]
    end

    K --> L["Raw Logits (7 values)"]
    L --> M["Softmax → Probabilities\n(7 class scores, sum=1)"]
    M --> N["argmax → Predicted Label\n+ Confidence Score"]

    N --> O

    subgraph APP ["🌐 Streamlit App (app.py)"]
        O["Result Card\n• Predicted label badge\n• Confidence %\n• Probability bars for all 7 classes"]
    end

    style PREPROCESS fill:#1e293b,stroke:#7c3aed,color:#e2e8f0
    style TOKENIZE fill:#1e293b,stroke:#4f46e5,color:#e2e8f0
    style MODEL fill:#1e293b,stroke:#0ea5e9,color:#e2e8f0
    style APP fill:#1e293b,stroke:#10b981,color:#e2e8f0
```

---

## 🗃️ File Structure & Responsibilities

```mermaid
graph LR
    subgraph PROJECT ["📁 Project Root"]
        CFG["config.py\n─────────────\n• MODEL_DIR path\n• MAX_LENGTH = 192\n• DEVICE = cpu\n• APP_TITLE / SUBTITLE\n• LABEL_COLORS dict"]

        PRE["preprocessing.py\n─────────────\n• clean_text()\n• Strip URLs\n• Strip @mentions\n• Normalise whitespace"]

        INF["inference.py\n─────────────\n• load_model_and_tokenizer()\n• predict(text, ...)\n• Returns label + confidence\n  + all probabilities"]

        APP["app.py\n─────────────\n• Streamlit UI\n• Hero section\n• Input text area\n• Result card\n• Probability bars\n• Disclaimer footer"]

        MDL["models/final_model/\n─────────────\n• model.safetensors (313 MB)\n• tokenizer.json (3.4 MB)\n• config.json\n• trainer_state.json\n• optimizer.pt"]

        DATA["data/\n─────────────\n• Combined Data.csv (~30 MB)\n  (training corpus)"]

        NB["Deep_Learning_Mental_Health\n_Sentiment_Analysis.ipynb\n─────────────\n• Full training notebook\n• Data EDA + preprocessing\n• Model experiments (v1-v5)\n• Evaluation + metrics"]
    end

    CFG --> INF
    CFG --> APP
    PRE --> INF
    INF --> APP
    MDL --> INF
    DATA --> NB
    NB --> MDL

    style CFG fill:#312e81,stroke:#818cf8,color:#e2e8f0
    style PRE fill:#1e3a5f,stroke:#38bdf8,color:#e2e8f0
    style INF fill:#1e3a5f,stroke:#38bdf8,color:#e2e8f0
    style APP fill:#14532d,stroke:#4ade80,color:#e2e8f0
    style MDL fill:#450a0a,stroke:#f87171,color:#e2e8f0
    style DATA fill:#431407,stroke:#fb923c,color:#e2e8f0
    style NB fill:#3b0764,stroke:#c084fc,color:#e2e8f0
```

---

## 🚀 Inference Pipeline Flow

```mermaid
sequenceDiagram
    actor User
    participant App as app.py (Streamlit)
    participant INF as inference.py
    participant PRE as preprocessing.py
    participant MDL as DistilRoBERTa Model

    User->>App: Types text & clicks "Analyze Text"
    App->>INF: predict(text, tokenizer, model, id2label)
    INF->>PRE: clean_text(raw_text)
    PRE-->>INF: cleaned_text (lowercase, no URLs, no @mentions)
    INF->>INF: tokenizer(cleaned_text, max_length=192)
    INF->>MDL: model(**inputs) [torch.no_grad()]
    MDL-->>INF: logits [shape: (1, 7)]
    INF->>INF: softmax(logits) → probabilities [shape: (7,)]
    INF->>INF: argmax → predicted_idx + confidence
    INF-->>App: {predicted_label, confidence, probabilities}
    App-->>User: Renders result card with badge + probability bars
```

---

## 🏋️ Training Pipeline (Notebook)

```mermaid
flowchart LR
    A["📦 Combined Data.csv\n~30 MB raw CSV"] --> B["🔍 EDA\n• Class distribution\n• Text length analysis\n• Sample inspection"]

    B --> C["🧹 Preprocessing\n• Lowercase\n• URL removal\n• @mention removal\n• Whitespace normalise"]

    C --> D["✂️ Train / Val Split\n• Stratified sampling\n• Balanced class weights"]

    D --> E["🔤 Tokenisation\n• DistilRoBERTa tokenizer\n• MAX_LEN = 192\n• Dynamic padding"]

    E --> F["🧠 Model Experiments\nv1 → v2 → v3 → v4 → v5\n(hyperparameter tuning)"]

    F --> G["🏆 Best Checkpoint (v5)\n• Step 1500\n• Epoch 2\n• F1 = 71.38%\n• Acc = 77.1%"]

    G --> H["💾 Saved Artefacts\nmodel.safetensors\ntokenizer.json\nconfig.json\ntrainer_state.json"]

    style A fill:#292524,stroke:#a16207,color:#fef9c3
    style G fill:#14532d,stroke:#4ade80,color:#d1fae5
    style H fill:#1e3a8a,stroke:#93c5fd,color:#dbeafe
```

---

## ⚙️ Technology Stack

```mermaid
graph TB
    subgraph ML ["🤖 Machine Learning"]
        T1["PyTorch ≥ 2.0"]
        T2["🤗 Transformers ≥ 4.40"]
        T3["SafeTensors ≥ 0.4"]
    end

    subgraph WEB ["🌐 Web App"]
        T4["Streamlit ≥ 1.35"]
        T5["HTML/CSS (custom)"]
        T6["Inter Google Font"]
    end

    subgraph MODEL_ARCH ["🧠 Model"]
        T7["DistilRoBERTa\n(RobertaForSequenceClassification)"]
        T8["HuggingFace AutoTokenizer"]
    end

    T1 --> T7
    T2 --> T7
    T2 --> T8
    T3 --> T7
    T4 --> T5

    style ML fill:#1e1b4b,stroke:#818cf8,color:#e2e8f0
    style WEB fill:#052e16,stroke:#4ade80,color:#e2e8f0
    style MODEL_ARCH fill:#1c1917,stroke:#a78bfa,color:#e2e8f0
```

---

## 🎨 UI & App Design Highlights

The Streamlit app features a **premium dark-mode design** with:

- 🌌 **Deep gradient background** — `#0f0c29 → #302b63 → #24243e`
- ✨ **Glassmorphism cards** — `rgba(255,255,255,0.05)` + `backdrop-filter: blur(10px)`
- 🎨 **Gradient hero title** — purple → blue → green text gradient
- 🏷️ **Color-coded prediction badges** — each class has its own unique colour
- 📊 **Animated probability bars** — smooth `0.6s ease` fill transitions
- 🔄 **FadeIn animation** — result cards slide up on appearance
- 🚫 **Medical disclaimer** — responsible AI, not a clinical tool

---

## 📁 Repository Layout

```
Transformer-Based Mental Health Classification/
│
├── 📓 Deep_Learning_Mental_Health_Sentiment_Analysis.ipynb  ← Full training notebook
├── 🚀 app.py              ← Streamlit web application
├── 🔧 config.py           ← Central configuration
├── 🤖 inference.py        ← Model loading + prediction logic
├── 🧹 preprocessing.py    ← Text cleaning pipeline
├── 📋 requirements.txt    ← Dependency manifest
├── 🚫 .gitignore
│
├── 📂 data/
│   └── Combined Data.csv  ← Training corpus (~30 MB)
│
└── 📂 models/
    ├── config.json         ← Top-level model config
    ├── distilroberta-v5-final.zip  ← Portable model archive (~696 MB)
    └── 📂 final_model/     ← Active model artefacts
        ├── model.safetensors     (313 MB — weights)
        ├── tokenizer.json        (3.4 MB)
        ├── tokenizer_config.json
        ├── config.json
        ├── trainer_state.json    ← Training history + metrics
        ├── training_args.bin
        ├── optimizer.pt          ← Optimizer state
        ├── scheduler.pt          ← LR scheduler state
        ├── scaler.pt             ← AMP scaler state
        └── rng_state.pth         ← RNG reproducibility seed
```

---

## 🔬 Key Engineering Decisions

| Decision | Rationale |
|---|---|
| **DistilRoBERTa over BERT** | 40% fewer parameters, 60% faster, 97% of BERT's performance |
| **MAX_LENGTH = 192** | Captures sufficient context while keeping inference fast |
| **Softmax over argmax** | Returns full probability distribution, not just the winner |
| **`@st.cache_resource`** | Model loads once per session — no re-loading on each request |
| **`local_files_only=True`** | Fully offline inference — no internet dependency at runtime |
| **`torch.no_grad()`** | Disables gradient computation during inference — saves memory & time |
| **SafeTensors format** | Safer and faster model serialisation vs. `.bin` pickle format |
| **Preprocessing parity** | `clean_text()` mirrors notebook Cell 6 exactly — no train/serve skew |

---

## 🏅 Final Achievement Summary

> ### 🎊 Congratulations on completing this project!

| Milestone | ✅ |
|---|---|
| Dataset collected and explored | ✅ |
| Preprocessing pipeline built | ✅ |
| Multiple model versions trained (v1→v5) | ✅ |
| Best checkpoint identified and saved | ✅ |
| Inference module built with full type hints | ✅ |
| Streamlit web app designed and deployed | ✅ |
| Premium dark-mode UI with animations | ✅ |
| Medical disclaimer and responsible AI notes | ✅ |
| Git repository with `.gitignore` | ✅ |
| `requirements.txt` for reproducibility | ✅ |

**Final Model Performance:**
- 🎯 **Accuracy: 77.1%** across 7 mental health categories
- 📐 **Macro F1: 71.38%** — strong balanced performance
- ⚡ **Inference: ~1s** on CPU with 192-token input

---

*This project demonstrates mastery of the full ML engineering lifecycle — from raw data to a deployed, user-facing application — using state-of-the-art transformer architectures in the mental health NLP domain.*

---
> ⚠️ **Disclaimer:** This tool is for research and portfolio purposes only. It is **not** a clinical diagnostic instrument. If you or someone you know is struggling, please seek professional support.
