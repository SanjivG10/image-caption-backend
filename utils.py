from PIL import Image

CAPTION_CATEGORIES = [
    "happy",
    "sarcastic",
    "sad",
    "exciting",
    "angry",
    "romantic",
    "nostalgic",
    "motivational",
]

import torch
from lavis.models import load_model_and_preprocess
import torch


def get_image_caption_from_ml(image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_image = Image.open(image).convert("RGB")
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip_caption", model_type="base_coco", is_eval=True, device=device
    )
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    caption = model.generate({"image": image})
    return caption


def check_image(request):
    f = request.files.get("image")
    try:
        img = Image.open(f)
        w, h = img.size
        if w > 50 and h > 50:
            return True
        return False
    except IOError:
        return False
