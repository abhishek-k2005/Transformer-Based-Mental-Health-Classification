"""
config.py
---------
Central configuration for the Mental Health Classification inference app.
All paths are relative to this file location so the project works after cloning.
"""

from pathlib import Path

# -- Root of the repository ---------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent

# -- Model artefacts ----------------------------------------------------------
# The winning DistilRoBERTa v5 checkpoint saved from the training notebook.
MODEL_DIR = ROOT_DIR / "models" / "final_model"

# -- Tokenisation settings (must match training) ------------------------------
# v5 used MAX_LEN_V5 = 192 with truncation=True, padding=False at batch time.
MAX_LENGTH: int = 192

# -- Inference device ---------------------------------------------------------
# "cpu" keeps the app portable; set "cuda" or "mps" if a GPU is present.
DEVICE: str = "cpu"

# -- UI / display -------------------------------------------------------------
APP_TITLE: str = "Mental Health Text Classifier"
APP_SUBTITLE: str = (
    "Powered by DistilRoBERTa · fine-tuned on the Combined Mental Health Dataset"
)

# Label colours used in the results panel (order matches id2label in config.json)
LABEL_COLORS: dict = {
    "Anxiety":              "#F59E0B",
    "Bipolar":              "#8B5CF6",
    "Depression":           "#6366F1",
    "Normal":               "#10B981",
    "Personality disorder": "#EC4899",
    "Stress":               "#F97316",
    "Suicidal":             "#EF4444",
}
