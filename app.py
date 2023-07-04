from flask import Flask, request
from flask import jsonify

# from neuraltalk import FeatureExtractor
from caption import get_model, get_captions
from flask_sqlalchemy import SQLAlchemy
from PIL import Image
import imagehash
import random
from gpt3 import generate_caption
from utils import CAPTION_CATEGORIES, check_image
from flask_cors import CORS, cross_origin

from lavis.models import model_zoo, load_model_and_preprocess
import torch
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
raw_image = Image.open("./girrafe.jpg").convert("RGB")
model, vis_processors, _ = load_model_and_preprocess(
    name="blip_caption", model_type="base_coco", is_eval=True, device=device
)
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
caption = model.generate({"image": image})
print(caption)


app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:////tmp/test.db"
db = SQLAlchemy(app)
CORS(app, origins="*")


class ImageCaption(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    hash = db.Column(db.Text)
    captions = db.Column(db.Text)


class AIImageCaption(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    caption = db.Column(db.Text)
    category = db.Column(db.Text)
    gen_caption = db.Column(db.Text)


MAX_LENGTH = 5


@cross_origin(origins="*")
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            category = random.choice(CAPTION_CATEGORIES).lower()
            category = request.form.get("category", CAPTION_CATEGORIES[0]).lower()
            if not category in CAPTION_CATEGORIES:
                return jsonify({"err": "Chosen category doesn't exist."}), 400
            is_valid = check_image(request)
            if not is_valid:
                return jsonify({"err": "bad request"}), 400

            image = request.files.get("image")
            hash = str(imagehash.average_hash(Image.open(image)))
            image_caption = ImageCaption.query.filter_by(hash=hash).first()

            captions = []
            if image_caption:
                captions = image_caption.captions
                captions = captions.split("-")
            else:
                # feature = feature_extractor(image)
                # captions = get_captions(model, feature)
                captions = []
                new_caption = ImageCaption(
                    hash=hash, captions="-".join(caption for caption in captions)
                )
                db.session.add(new_caption)
                db.session.commit()

            if len(captions) == 0:
                return jsonify({"err": "couldn't infer anything from the image"}), 500

            random_caption = random.choice(captions)
            ai_generated_caption = generate_caption(random_caption, category)

            ai_caption = AIImageCaption(
                category=category,
                caption=random_caption,
                gen_caption=ai_generated_caption,
            )
            db.session.add(ai_caption)
            db.session.commit()

            return jsonify({"caption": ai_generated_caption[0]})

        except Exception as e:
            print(e)
            return jsonify({"err": str(e)}), 500

    return jsonify({"err": "METHOD is not supported"}), 405
