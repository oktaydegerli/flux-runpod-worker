import torch
import shutil
from pathlib import Path
from diffusers import DiffusionPipeline

MODEL_ID = "black-forest-labs/FLUX.1-schnell"
MODEL_CACHE_DIR = "/workspace/diffusers-cache"

model_cache_path = Path(MODEL_CACHE_DIR)
if model_cache_path.exists():
    shutil.rmtree(model_cache_path)
model_cache_path.mkdir(parents=True, exist_ok=True)

DiffusionPipeline.from_pretrained(
    MODEL_ID,
    cache_dir=Path(MODEL_CACHE_DIR),
    torch_dtype=torch.bfloat16
).to('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
