import os
import torch
import numpy as np
import logging
import base64
import requests
from cog import BasePredictor, Input, Secret, Path
from PIL import Image
from diffusers import (
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLAdapterPipeline,
    T2IAdapter,
    AutoencoderKL,
    EulerAncestralDiscreteScheduler,
)
from diffusers.utils import load_image as load_pil_image
from openai import OpenAI
from segment_anything import SamPredictor, sam_model_registry
from scipy.ndimage import binary_dilation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_MODEL_ID = "madebyollin/sdxl-vae-fp16-fix"
ADAPTER_MODEL_ID = "TencentARC/t2i-adapter-depth-midas-sdxl-1.0"
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
SAM_TYPE = "vit_h"
SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

SCHEDULERS_MAP = {"K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler}


def download_sam_checkpoint():
    if not os.path.exists(SAM_CHECKPOINT):
        logger.info(f"Downloading SAM checkpoint from {SAM_CHECKPOINT_URL}...")
        response = requests.get(SAM_CHECKPOINT_URL, stream=True)
        response.raise_for_status()
        with open(SAM_CHECKPOINT, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info("SAM checkpoint downloaded.")


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
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                ]
            }
        ],
        max_tokens=50
    )
    coords = response.choices[0].message.content.strip()
    if coords.lower() == "none":
        raise ValueError("GPT-4 Vision could not identify requested features.")
    # validate format
    for pt in coords.split(";"):
        _ = map(int, pt.strip().split(","))
    return coords


def validate_points(points_str):
    try:
        for pt in points_str.split(";"):
            x, y = map(int, pt.strip().split(","))
            if x < 0 or y < 0:
                raise ValueError
        return True
    except:
        return False


def _prepare_depth_map(pil_image, width, height):
    # Placeholder: implement your depth estimator here if needed
    # For this pipeline, we assume depth adapter handles raw image
    return pil_image.resize((width, height), Image.Resampling.LANCZOS)

class Predictor(BasePredictor):
    def setup(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16

        # Load Inpaint Pipeline
        logger.info("Loading SDXL Inpaint pipeline…")
        self.inpainter = StableDiffusionXLInpaintPipeline.from_pretrained(
            BASE_MODEL_ID, torch_dtype=self.dtype
        ).to(self.device)

        # Load Depth-Adapter Pipeline
        logger.info("Loading SDXL Adapter pipeline…")
        vae = AutoencoderKL.from_pretrained(VAE_MODEL_ID, torch_dtype=self.dtype)
        adapter = T2IAdapter.from_pretrained(ADAPTER_MODEL_ID, torch_dtype=self.dtype)
        self.depth_pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
            BASE_MODEL_ID, vae=vae, adapter=adapter, torch_dtype=self.dtype
        ).to(self.device)
        self.depth_pipe.scheduler = SCHEDULERS_MAP["K_EULER_ANCESTRAL"].from_config(
            self.depth_pipe.scheduler.config
        )
        self.depth_pipe.enable_vae_tiling()
        self.depth_pipe.enable_model_cpu_offload()
        try:
            self.depth_pipe.enable_xformers_memory_efficient_attention()
        except:
            self.depth_pipe.enable_attention_slicing()

        # Load SAM
        download_sam_checkpoint()
        sam = sam_model_registry[SAM_TYPE](checkpoint=SAM_CHECKPOINT)
        self.sam = SamPredictor(sam)

    def predict(
        self,
        prompt: str = Input(description="Final polish prompt"),
        image: Path = Input(description="Input photo"),
        mask_target: str = Input(
            default=None,
            description="Target(s) for SAM, e.g. 'front door'"
        ),
        mask_points: str = Input(
            default=None,
            description="Fallback manual points 'x1,y1;x2,y2'"
        ),
        openai_api_key: Secret = Input(
            default=None,
            description="OpenAI API key for GPT-Vision"
        ),
        seed: int = Input(default=None),
    ) -> Path:
        seed = seed or int.from_bytes(os.urandom(4), "big")
        gen = torch.Generator(device=self.device).manual_seed(seed)

        pil = load_pil_image(str(image)).convert("RGB")
        np_img = np.array(pil)

        # Determine door center points
        points = []
        if mask_target and openai_api_key:
            api_key = openai_api_key.get_secret_value()
            coords = gpt_vision_find_points(str(image), mask_target, api_key)
            for pt in coords.split(";"):
                x, y = map(int, pt.strip().split(","))
                points.append([x, y])
        elif mask_points and validate_points(mask_points):
            for pt in mask_points.split(";"):
                x, y = map(int, pt.strip().split(","))
                points.append([x, y])
        else:
            raise ValueError("Provide either mask_target+key or mask_points.")

        # Generate SAM masks
        self.sam.set_image(np_img)
        masks, _, _ = self.sam.predict(
            point_coords=np.array(points),
            point_labels=np.ones(len(points)),
            multimask_output=False
        )
        door_mask = masks[0]
        wall_mask = ~door_mask

        door_mask_img = Image.fromarray((door_mask * 255).astype("uint8"))
        wall_mask_img = Image.fromarray((wall_mask * 255).astype("uint8"))

        # Inpaint door to blue
        blue_door = self.inpainter(
            prompt="A bright blue front door, photorealistic.",
            image=pil,
            mask_image=door_mask_img,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=gen,
        ).images[0]

        # Inpaint walls to orange
        orange_house = self.inpainter(
            prompt="Recolor the house exterior a warm orange shade, photorealistic.",
            image=blue_door,
            mask_image=wall_mask_img,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=gen,
        ).images[0]

        # Depth-polish final image
        depth_map = _prepare_depth_map(pil, orange_house.width, orange_house.height)
        final = self.depth_pipe(
            prompt=prompt,
            image=depth_map,
            num_inference_steps=20,
            guidance_scale=1.0,
            adapter_conditioning_scale=0.3,
            generator=gen,
        ).images[0]

        out_path = "/tmp/out.png"
        final.save(out_path)
        return Path(out_path)

