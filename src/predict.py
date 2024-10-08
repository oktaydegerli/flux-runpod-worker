import torch
from diffusers import DiffusionPipeline
import os
import shutil
from typing import List
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image

MODEL_ID = "black-forest-labs/FLUX.1-schnell"
MODEL_CACHE_DIR = "/runpod-volume/diffusers-cache/flux-schnell"

class Predictor:
    def setup(self):
        print("Loading pipeline...")

        model_fetched = True
        model_cache_path = Path(MODEL_CACHE_DIR)
        
        if not model_cache_path.exists():
            model_cache_path.mkdir(parents=True, exist_ok=True)
            model_fetched = False
    
        self.pipe = DiffusionPipeline.from_pretrained(
            MODEL_ID,
            cache_dir=Path(MODEL_CACHE_DIR),
            local_files_only=model_fetched,
            torch_dtype=torch.bfloat16
        ).to('cuda')

    @torch.inference_mode()
    def predict(self, prompt, width, height, num_outputs, num_inference_steps, guidance_scale, seed):
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if width * height > 1048576:
            raise ValueError(
                "Maximum size is 1024x024 pixels, because of memory limits. Please select a lower width or height."
            )

        generator = torch.Generator("cuda").manual_seed(seed)

        output = self.pipe(
            prompt=[prompt] * num_outputs if prompt is not None else None,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pil",
            num_inference_steps=num_inference_steps,
        )

        outputs = []

        for i, img in enumerate(output.images):
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            outputs.append(img_b64)

        return outputs
