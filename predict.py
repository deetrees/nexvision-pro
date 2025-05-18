import os
import torch
import base64
import openai
from PIL import Image as PILImage
from cog import BasePredictor, Input, Path, Secret
from diffusers import StableDiffusionInpaintPipeline
from sam2 import Sam2Predictor

def gpt_vision_find_point(image_path, target, api_key):
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    prompt = f"Given this image of a house, what are the pixel coordinates (x, y) of the center of the {target}? Respond ONLY with the coordinates as: x,y"
    response = openai.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {"role": "system", "content": "You are an expert at interpreting house images."},
            {"role": "user",
             "content": [
                 {"type": "text", "text": prompt},
                 {"type": "image_url", "image_url": f"data:image/png;base64,{img_b64}"}
             ]
            }
        ],
        max_tokens=20,
        api_key=api_key
    )
    coords = response.choices[0].message.content.strip()
    return coords

class Predictor(BasePredictor):
    def setup(self):
        # Load your inpaint pipeline/model here (customize as needed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            torch_dtype=torch.float16
        ).to(self.device)
        if self.device.type == 'cuda':
            self.inpaint_pipe.enable_model_cpu_offload()
            try:
                self.inpaint_pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                self.inpaint_pipe.enable_attention_slicing()
from segment_anything import SamPredictor, sam_model_registry

    def predict(
        self,
        prompt: str = Input(description="Edit prompt, e.g. 'make the front door blue'"),
        image: Path = Input(description="Input image"),
        mask_target: str = Input(description="What to edit (e.g., 'front door', 'garage door')"),
        openai_api_key: Secret = Input(description="OpenAI API key"),
        num_inference_steps: int = Input(default=30, ge=1, le=100),
        guidance_scale: float = Input(default=7.5, ge=1.0, le=20.0),
        seed: int = Input(default=None, description="Random seed")
    ) -> Path:
        # Step 1: Get target point with GPT Vision
        coords = gpt_vision_find_point(str(image), mask_target, openai_api_key.get_secret_value())
        x, y = [int(i) for i in coords.split(",")]
        # Step 2: Generate mask with SAM-2
        pil_image = PILImage.open(str(image)).convert("RGB")
        points = torch.tensor([[x, y]])
        masks = self.sam2_predictor.predict(image=pil_image, points=points)
        mask_img = PILImage.fromarray(masks[0] * 255).convert("L")
        # Optional: Resize mask to match input if needed

        # Step 3: Inpaint with SDXL
        generator = torch.manual_seed(seed) if seed is not None else None
        result = self.inpaint_pipe(
            prompt=prompt,
            image=pil_image,
            mask_image=mask_img,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )
        result_image = result.images[0]
        out_path = "/tmp/out.png"
        result_image.save(out_path)
        return Path(out_path)

