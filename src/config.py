from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS = PROJECT_ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
MODELS = OUTPUTS / "models"

EMBEDDING_DIM = 100
BASELINE_MODEL = "all-MiniLM-L6-v2"
RANDOM_STATE = 42

DOC_TYPE_LABELS = {
    "20240809133545974_23-980_petbrief.pdf": "petition_brief",
    "22-1165_10n2.pdf": "slip_opinion",
    "23-980bsacus_facebookvamalgamatedbank.pdf": "amicus_brief",
    "USCOURTS-ilnd-1_04-cv-00397-5.pdf": "district_opinion",
    "comp26286.pdf": "sec_filing",
}

MIN_PARAGRAPH_CHARS = 50
