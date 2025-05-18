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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- USER CONFIGURATION ---
BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_MODEL_ID = "madebyollin/sdxl-vae-fp16-fix"
ADAPTER_MODEL_ID = "TencentARC/t2i-adapter-depth-midas-sdxl-1.0"
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
SAM_TYPE = "vit_h"
SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

SCHEDULERS_MAP = {
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
}

def download_sam_checkpoint():
    """Download SAM checkpoint if not present"""
    if not os.path.exists(SAM_CHECKPOINT):
        logger.info(f"Downloading SAM checkpoint from {SAM_CHECKPOINT_URL}...")
        try:
            response = requests.get(SAM_CHECKPOINT_URL, stream=True)
            response.raise_for_status()
            with open(SAM_CHECKPOINT, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info("SAM checkpoint downloaded.")
        except Exception as e:
            logger.error(f"Failed to download SAM checkpoint: {e}")
            raise

def gpt_vision_find_points(image_path, target, api_key):
    """Use GPT-4 Vision to find coordinates for target(s)"""
    try:
        logger.info(f"Finding coordinates for {target} with GPT-4 Vision...")
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        prompt = f"Given this image of a house, provide pixel coordinates (x,y) for the center of each {target} (e.g., 'front door, garage door'). Respond as: x1,y1;x2,y2"
        # FIX: Do NOT pass proxies or any extra arguments to OpenAI here!
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {"role": "system", "content": "You are an expert at interpreting house images."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": f"data:image/png;base64,{img_b64}"}
                    ]
                }
            ],
            max_tokens=50
        )
        coords = response.choices[0].message.content.strip()
        if not coords or ";" not in coords and "," not in coords:
            raise ValueError("Invalid coordinates format from GPT-4 Vision")
        logger.info(f"Coordinates found: {coords}")
        return coords
    except Exception as e:
        logger.error(f"GPT-4 Vision failed for {target}: {e}")
        raise

def validate_points(points_str):
    """Validate mask_points format (e.g., 'x1,y1;x2,y2')"""
    try:
        for pt in points_str.split(";"):
            x, y = map(int, pt.strip().split(","))
            if x < 0 or y < 0:
                raise ValueError("Coordinates must be non-negative")
        return True
    except Exception as e:
        logger.error(f"Invalid mask_points format: {e}")
        return False

class Predictor(BasePredictor):
    def setup(self):
        """Load all models into memory"""
        try:
            logger.info("Starting setup...")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.dtype = torch.float16
            logger.info(f"Using device: {self.device}")

            # Download SAM checkpoint
            download_sam_checkpoint()

            # Load SDXL pipeline
            logger.info("Loading SDXL pipeline...")
            vae = AutoencoderKL.from_pretrained(VAE_MODEL_ID, torch_dtype=self.dtype)
            adapter = T2IAdapter.from_pretrained(ADAPTER_MODEL_ID, torch_dtype=self.dtype)
            self.pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
                BASE_MODEL_ID,
                vae=vae,
                adapter=adapter,
                torch_dtype=self.dtype,
                use_safetensors=True,
            )
            self.pipe.to(self.device)
            self.pipe.scheduler = SCHEDULERS_MAP["K_EULER_ANCESTRAL"].from_config(self.pipe.scheduler.config)
            self.pipe.enable_vae_tiling()
            if self.device.type == "cuda":
                self.pipe.enable_model_cpu_offload()
                self.pipe.enable_sequential_cpu_offload()
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    logger.warning("xformers not available, using attention slicing")
                    self.pipe.enable_attention_slicing()
            logger.info("SDXL pipeline loaded.")

            # Load SAM
            logger.info("Loading SAM model...")
            sam = sam_model_registry[SAM_TYPE](checkpoint=SAM_CHECKPOINT)
            self.sam_predictor = SamPredictor(sam)
            logger.info("SAM loaded.")
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            raise

    def predict(
        self,
        prompt: str = Input(description="Prompt for SDXL, e.g., 'change front door to green'"),
        image: Path = Input(description="Input image for SDXL and SAM segmentation"),
        negative_prompt: str = Input(
            description="Negative prompt",
            default="ugly, deformed, noisy, blurry, distorted, out of focus, bad anatomy, extra limbs, poorly drawn face, poorly drawn hands, missing fingers, low quality, jpeg artifacts, worst quality, watermark, signature",
        ),
        num_inference_steps: int = Input(default=20, ge=10, le=50, description="Number of denoising steps"),
        guidance_scale: float = Input(default=7.5, ge=1.0, le=20.0, description="Classifier-free guidance scale"),
        adapter_conditioning_scale: float = Input(default=0.9, ge=0.0, le=1.0, description="Adapter conditioning scale"),
        seed: int = Input(default=None, description="Random seed for reproducibility"),
        mask_target: str = Input(
            description="Parts to edit (e.g., 'front door, garage door'). Requires OpenAI API key.",
            default=None,
        ),
        mask_points: str = Input(
            description="Comma-separated points for SAM mask (e.g., 'x1,y1;x2,y2'). Ignored if mask_target provided.",
            default=None,
        ),
        openai_api_key: Secret = Input(description="OpenAI API key for GPT-4 Vision", default=None),
    ) -> Path:
        try:
            logger.info("Starting prediction...")

            # Validate inputs
            if not prompt:
                raise ValueError("Prompt is required")
            if not os.path.exists(str(image)):
                raise ValueError("Input image not found")

            # Set seed
            if seed is None:
                seed = int.from_bytes(os.urandom(4), "big")
            generator = torch.Generator(device=self.device).manual_seed(seed)
            logger.info(f"Using seed: {seed}")

            # Load and prep input image
            input_image = load_pil_image(str(image)).convert("RGB")
            np_image = np.array(input_image)

            # Get mask points
            points = []
            if mask_target and openai_api_key:
                api_key = openai_api_key.get_secret_value() or os.environ.get("REPLICATE_OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OpenAI API key not provided")
                coords = gpt_vision_find_points(str(image), mask_target, api_key)
                for pt in coords.split(";"):
                    x, y = map(int, pt.strip().split(","))
                    if x < 0 or y < 0:
                        raise ValueError("Coordinates must be non-negative")
                    points.append([x, y])
            elif mask_points:
                if not validate_points(mask_points):
                    raise ValueError("Invalid mask_points format")
                for pt in mask_points.split(";"):
                    x, y = map(int, pt.strip().split(","))
                    points.append([x, y])
            else:
                logger.warning("No mask_target or mask_points provided; processing entire image.")

            # Generate mask with SAM
            mask = None
            if points:
                try:
                    points_np = np.array(points)
                    self.sam_predictor.set_image(np_image)
                    masks, scores, _ = self.sam_predictor.predict(
                        point_coords=points_np,
                        point_labels=np.ones(len(points)),
                        multimask_output=False,
                    )
                    # Combine masks for multiple targets
                    mask = np.any(masks, axis=0)
                    # Dilate mask for better coverage
                    mask = binary_dilation(mask, iterations=5)
                    # Save mask for debugging
                    Image.fromarray((mask * 255).astype("uint8")).save("/tmp/sam_mask.png")
                except Exception as e:
                    logger.error(f"SAM mask generation failed: {e}")
                    mask = None

            # Prepare for SDXL (use original image, as T2I-Adapter uses depth)
            control_image = input_image

            # Generate with SDXL Pipeline
            logger.info("Running SDXL pipeline...")
            with torch.no_grad():
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=control_image,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    adapter_conditioning_scale=adapter_conditioning_scale,
                    generator=generator,
                ).images[0]

            out_path = "/tmp/out.png"
            result.save(out_path)
            logger.info("Prediction complete.")
            return Path(out_path)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

