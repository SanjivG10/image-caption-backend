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
import requests
from PIL import Image
from io import BytesIO


def open_image_from_url(url):
    response = requests.get(url)
    image_data = response.content
    image = Image.open(BytesIO(image_data)).convert("RGB")
    return image


def get_image_caption_from_ml(image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Opening Raw Image")
    raw_image = open_image_from_url(image)
    print("Raw Image Downloaded and loaded")
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
