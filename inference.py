"""
inference.py
------------
Model / tokeniser loading and prediction utilities for the
DistilRoBERTa v5 mental-health classifier.

Usage
-----
    from inference import load_model_and_tokenizer, predict

    tokenizer, model, id2label = load_model_and_tokenizer()
    result = predict("I have been feeling really anxious lately.", tokenizer, model, id2label)
    print(result)
    # {
    #   "predicted_label": "Anxiety",
    #   "confidence": 0.87,
    #   "probabilities": {"Anxiety": 0.87, "Bipolar": 0.02, ...}
    # }
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import DEVICE, MAX_LENGTH, MODEL_DIR
from preprocessing import clean_text


def load_model_and_tokenizer(
    model_dir: str | None = None,
    device: str | None = None,
) -> tuple[Any, Any, dict[int, str]]:
    """
    Load the fine-tuned DistilRoBERTa tokeniser and model from disk.

    Parameters
    ----------
    model_dir : str or Path, optional
        Path to the saved model directory.  Defaults to ``config.MODEL_DIR``.
    device : str, optional
        Torch device string ("cpu", "cuda", "mps").  Defaults to ``config.DEVICE``.

    Returns
    -------
    tokenizer : PreTrainedTokenizer
    model : PreTrainedModel   (in eval mode, on the requested device)
    id2label : dict[int, str]
        Integer label index -> human-readable class name, read from the model
        configuration so we never need to hard-code labels here.
    """
    model_dir = str(model_dir or MODEL_DIR)
    device = device or DEVICE

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    model.to(device)
    model.eval()

    # Read class labels from the saved model configuration.
    raw_id2label: dict = model.config.id2label
    id2label: dict[int, str] = {int(k): v for k, v in raw_id2label.items()}

    return tokenizer, model, id2label


def predict(
    text: str,
    tokenizer: Any,
    model: Any,
    id2label: dict[int, str],
    device: str | None = None,
    max_length: int = MAX_LENGTH,
) -> dict[str, Any]:
    """
    Run inference on a single text string.

    The input is cleaned with the same ``clean_text`` function used during
    training before being tokenised.

    Parameters
    ----------
    text : str
        Raw user-supplied text.
    tokenizer : PreTrainedTokenizer
    model : PreTrainedModel
    id2label : dict[int, str]
    device : str, optional
    max_length : int
        Token truncation length (192, matching v5 training).

    Returns
    -------
    dict with keys:
        predicted_label : str   -- the winning class name
        confidence      : float -- probability of the winning class (0-1)
        probabilities   : dict[str, float] -- softmax probability for every class
    """
    device = device or DEVICE

    cleaned = clean_text(text)

    inputs = tokenizer(
        cleaned,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits  # shape (1, num_labels)
    probs = F.softmax(logits, dim=-1).squeeze(0)  # shape (num_labels,)

    predicted_idx = int(probs.argmax().item())
    confidence = float(probs[predicted_idx].item())

    num_labels = probs.shape[0]
    probabilities: dict[str, float] = {
        id2label[i]: float(probs[i].item()) for i in range(num_labels)
    }

    return {
        "predicted_label": id2label[predicted_idx],
        "confidence": confidence,
        "probabilities": probabilities,
    }
