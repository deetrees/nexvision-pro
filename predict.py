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
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    AutoencoderKL,
    EulerAncestralDiscreteScheduler,
)
from diffusers.utils import load_image as load_pil_image
from openai import OpenAI
from segment_anything import SamPredictor, sam_model_registry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_MODEL_ID = "madebyollin/sdxl-vae-fp16-fix"
CONTROLNET_DEPTH_ID = "diffusers/controlnet-depth-sdxl-1.0"
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
    # Validate format
    for pt in coords.split(";"):
        if pt.strip():
            _ = map(int, pt.strip().split(","))
    return coords


def validate_points(points_str):
    try:
        for pt in points_str.split(";"):
            pt = pt.strip()
            if not pt:
                continue
            x, y = map(int, pt.strip().split(","))
            if x < 0 or y < 0:
                raise ValueError
        return True
    except:
        return False


def _prepare_depth_map(pil_image, width, height):
    return pil_image.resize((width, height), Image.Resampling.LANCZOS)


class Predictor(BasePredictor):
    def setup(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16

        # Inpainting
        logger.info("Loading SDXL Inpaint pipeline…")
        self.inpainter = StableDiffusionXLInpaintPipeline.from_pretrained(
            BASE_MODEL_ID, torch_dtype=self.dtype
        ).to(self.device)

        # ControlNet
        logger.info("Loading SDXL ControlNet Depth pipeline…")
        vae = AutoencoderKL.from_pretrained(VAE_MODEL_ID, torch_dtype=self.dtype)
        controlnet = ControlNetModel.from_pretrained(CONTROLNET_DEPTH_ID, torch_dtype=self.dtype)
        self.control_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            BASE_MODEL_ID, vae=vae, controlnet=controlnet, torch_dtype=self.dtype
        ).to(self.device)
        self.control_pipe.scheduler = SCHEDULERS_MAP["K_EULER_ANCESTRAL"].from_config(
            self.control_pipe.scheduler.config
        )
        self.control_pipe.enable_vae_tiling()
        self.control_pipe.enable_model_cpu_offload()
        try:
            self.control_pipe.enable_xformers_memory_efficient_attention()
        except:
            self.control_pipe.enable_attention_slicing()

        # Load SAM
        download_sam_checkpoint()
        sam = sam_model_registry[SAM_TYPE](checkpoint=SAM_CHECKPOINT)
        self.sam = SamPredictor(sam)

    def predict(
        self,
        prompt: str = Input(description="Describe the whole remodel, e.g. 'a house with red siding and a blue front door, photorealistic'"),
        image: Path = Input(description="Input house photo"),
        house_color: str = Input(default="red", description="Color for the house exterior"),
        door_color: str = Input(default="blue", description="Color for the door"),
        mask_targets: str = Input(default="house exterior;front door", description="Semicolon-separated targets for mask: 'house exterior;front door'"),
        openai_api_key: Secret = Input(default=None, description="OpenAI API key for GPT-Vision"),
        num_inference_steps: int = Input(default=30, description="Inference steps for inpainting and ControlNet"),
        guidance_scale: float = Input(default=7.5, description="Prompt guidance scale (higher = stricter prompt)"),
        controlnet_conditioning_scale: float = Input(default=0.7, description="ControlNet structure fidelity"),
        negative_prompt: str = Input(default="cartoon, illustration, painting, text, watermark", description="Negative prompt to enforce realism"),
        seed: int = Input(default=None, description="Random seed"),
    ) -> Path:
        seed = seed or int.from_bytes(os.urandom(4), "big")
        gen = torch.Generator(device=self.device).manual_seed(seed)
        pil = load_pil_image(str(image)).convert("RGB")
        np_img = np.array(pil)

        # ---- Generate masks for each target ----
        targets = [t.strip() for t in mask_targets.split(";")]
        colors = [house_color, door_color]  # Adapt this to add more regions
        region_prompts = [
            f"a {c} {t}, photorealistic"
            for c, t in zip(colors, targets)
        ]

        current_image = pil
        api_key = openai_api_key.get_secret_value() if openai_api_key else None

        for target, region_prompt in zip(targets, region_prompts):
            # Get points with GPT-4 Vision or fallback
            coords = None
            if api_key:
                try:
                    coords = gpt_vision_find_points(str(image), target, api_key)
                except Exception as e:
                    logger.warning(f"GPT-4 Vision failed for {target}: {e}")
            if not coords:
                raise ValueError(f"No mask points for target: {target}")

            # Robust parsing of points (ignore blanks, skip invalid)
            points = []
            for pt in coords.split(";"):
                pt = pt.strip()
                if not pt:
                    continue
                try:
                    x, y = map(int, pt.split(","))
                    points.append([x, y])
                except Exception as e:
                    logger.warning(f"Skipping invalid point '{pt}': {e}")

            if not points:
                raise ValueError(f"No valid points found for target: {target}")

            # Segment mask with SAM
            self.sam.set_image(np.array(current_image))
            masks, _, _ = self.sam.predict(
                point_coords=np.array(points),
                point_labels=np.ones(len(points)),
                multimask_output=False
            )
            region_mask = masks[0]
            region_mask_img = Image.fromarray((region_mask * 255).astype("uint8"))

            # Inpaint the region
            logger.info(f"Inpainting {target} with prompt: {region_prompt}")
            result = self.inpainter(
                prompt=region_prompt,
                image=current_image,
                mask_image=region_mask_img,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                generator=gen,
            )
            current_image = result.images[0]

        # ---- Final ControlNet Depth Polish ----
        logger.info("Running ControlNet polish for structure...")
        depth_map = _prepare_depth_map(pil, current_image.width, current_image.height)
        final = self.control_pipe(
            prompt=prompt,
            image=current_image,
            control_image=depth_map,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            negative_prompt=negative_prompt,
            generator=gen,
        ).images[0]

        out_path = "/tmp/out.png"
        final.save(out_path)
        return Path(out_path)

