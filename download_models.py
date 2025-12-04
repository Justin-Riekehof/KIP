# download_models.py
import os
from huggingface_hub import snapshot_download

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Whisper (faster-whisper small)
snapshot_download(
    repo_id="Systran/faster-whisper-small",
    local_dir=os.path.join(MODELS_DIR, "whisper", "faster-whisper-small"),
    local_dir_use_symlinks=False,  # wichtig für Windows / Git
)

# Pyannote community diarization pipeline
# HUGGINGFACE_TOKEN vorher als Umgebungsvariable setzen!
snapshot_download(
    repo_id="pyannote/speaker-diarization-community-1",
    local_dir=os.path.join(MODELS_DIR, "pyannote", "speaker-diarization-community-1"),
    local_dir_use_symlinks=False,
    token=os.environ.get("HUGGINGFACE_TOKEN"),
)
