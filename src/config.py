from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# Experiment output directory (1 = Word2Vec vs plain MiniLM, 2 = plain vs finetuned MiniLM)
RUN_ID = "1"
OUTPUTS = PROJECT_ROOT / "outputs" / RUN_ID
FIGURES = OUTPUTS / "figures"
MODELS = OUTPUTS / "models"
FINETUNED_MODEL_DIR = MODELS / "finetuned_minilm"


def set_run_id(run_id: str) -> None:
    """Point pipeline outputs at outputs/<run_id>/. Call before importing stage modules."""
    global RUN_ID, OUTPUTS, FIGURES, MODELS, FINETUNED_MODEL_DIR
    RUN_ID = run_id
    OUTPUTS = PROJECT_ROOT / "outputs" / RUN_ID
    FIGURES = OUTPUTS / "figures"
    MODELS = OUTPUTS / "models"
    FINETUNED_MODEL_DIR = MODELS / "finetuned_minilm"

EMBEDDING_DIM = 100
BASELINE_MODEL = "all-MiniLM-L6-v2"
RANDOM_STATE = 42

# Fine-tuning hyperparameters
FINETUNE_EPOCHS = 3
FINETUNE_BATCH_SIZE = 16
FINETUNE_WARMUP_STEPS = 100
FINETUNE_MAX_PAIRS_PER_DOC = 400

DOC_TYPE_LABELS = {
    "20240809133545974_23-980_petbrief.pdf": "petition_brief",
    "22-1165_10n2.pdf": "slip_opinion",
    "23-980bsacus_facebookvamalgamatedbank.pdf": "amicus_brief",
    "USCOURTS-ilnd-1_04-cv-00397-5.pdf": "district_opinion",
    "comp26286.pdf": "sec_filing",
}

MIN_PARAGRAPH_CHARS = 50
