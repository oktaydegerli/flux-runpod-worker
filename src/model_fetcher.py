import torch
import shutil
from pathlib import Path
from diffusers import DiffusionPipeline

MODEL_ID = "black-forest-labs/FLUX.1-schnell"
MODEL_CACHE_DIR = "/workspace/diffusers-cache"

model_cache_path = Path(MODEL_CACHE_DIR)
if not model_cache_path.exists():
    DiffusionPipeline.from_pretrained(
        MODEL_ID,
        cache_dir=Path(MODEL_CACHE_DIR),
        torch_dtype=torch.bfloat16
    ).to('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
