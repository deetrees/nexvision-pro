import os
import torch
import numpy as np
import logging
import base64
from cog import BasePredictor, Input, Path, Secret
from PIL import Image
from diffusers import (
    StableDiffusionXLAdapterPipeline,
    T2IAdapter,
    AutoencoderKL,
    EulerAncestralDiscreteScheduler,
)
from diffusers.utils import load_image as load_pil_image
from openai import OpenAI
from segment_anything import SamPredictor, sam_model_registry
import requests
from scipy.ndimage import binary_dilation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_MODEL_ID = "madebyollin/sdxl-vae-fp16-fix"
ADAPTER_MODEL_ID = "TencentARC/t2i-adapter-depth-midas-sdxl-1.0"
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
SAM_TYPE = "vit_h"
SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

SCHEDULERS_MAP = {"K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler}

def download_sam_checkpoint():
    if not os.path.exists(SAM_CHECKPOINT):
        response = requests.get(SAM_CHECKPOINT_URL, stream=True)
        response.raise_for_status()
        with open(SAM_CHECKPOINT, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

def gpt_vision_find_points(image_path, target, api_key):
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    prompt = (
        f"Given this image of a house, identify exactly: {target}. "
        "Provide only numeric pixel coordinates (x,y) for each feature. "
        "Respond as x1,y1;x2,y2. If unsure, say 'None'. No explanations."
    )

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Respond only with numeric coordinates or 'None'."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": f"data:image/png;base64,{img_b64}"}
            ]}
        ],
        max_tokens=50
    )

    coords = response.choices[0].message.content.strip()
    if coords.lower() == "none":
        raise ValueError("GPT-4 Vision could not identify requested features.")

    for pt in coords.split(";"):
        x, y = map(int, pt.strip().split(","))

    return coords

class Predictor(BasePredictor):
    def setup(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16
        download_sam_checkpoint()

        vae = AutoencoderKL.from_pretrained(VAE_MODEL_ID, torch_dtype=self.dtype)
        adapter = T2IAdapter.from_pretrained(ADAPTER_MODEL_ID, torch_dtype=self.dtype)

        self.pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
            BASE_MODEL_ID, vae=vae, adapter=adapter, torch_dtype=self.dtype, use_safetensors=True
        ).to(self.device)

        self.pipe.scheduler = SCHEDULERS_MAP["K_EULER_ANCESTRAL"].from_config(self.pipe.scheduler.config)
        self.pipe.enable_vae_tiling()

        if self.device.type == "cuda":
            self.pipe.enable_model_cpu_offload()
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                self.pipe.enable_attention_slicing()

        sam = sam_model_registry[SAM_TYPE](checkpoint=SAM_CHECKPOINT)
        self.sam_predictor = SamPredictor(sam)

    def predict(
        self,
        prompt: str,
        image: Path,
        negative_prompt: str = Input(default="ugly, distorted, bad anatomy"),
        num_inference_steps: int = Input(default=20, ge=10, le=50),
        guidance_scale: float = Input(default=7.5, ge=1.0, le=20.0),
        adapter_conditioning_scale: float = Input(default=0.9, ge=0.0, le=1.0),
        seed: int = Input(default=None),
        mask_target: str = Input(default=None),
        mask_points: str = Input(default=None),
        openai_api_key: Secret = Input(default=None),
    ) -> Path:
        seed = seed or int.from_bytes(os.urandom(4), "big")
        generator = torch.Generator(device=self.device).manual_seed(seed)

        input_image = load_pil_image(str(image)).convert("RGB")
        np_image = np.array(input_image)

        points = []
        if mask_target and openai_api_key:
            api_key = openai_api_key.get_secret_value()
            coords = gpt_vision_find_points(str(image), mask_target, api_key)
            for pt in coords.split(";"):
                x, y = map(int, pt.strip().split(","))
                points.append([x, y])

        mask = None
        if points:
            points_np = np.array(points)
            self.sam_predictor.set_image(np_image)
            masks, _, _ = self.sam_predictor.predict(
                point_coords=points_np,
                point_labels=np.ones(len(points)),
                multimask_output=False,
            )
            mask = binary_dilation(np.any(masks, axis=0), iterations=5)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=input_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            adapter_conditioning_scale=adapter_conditioning_scale,
            generator=generator,
        ).images[0]

        out_path = "/tmp/out.png"
        result.save(out_path)
        return Path(out_path)

