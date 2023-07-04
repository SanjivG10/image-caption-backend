import datetime
import random

import boto3
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from flask_sqlalchemy import SQLAlchemy
from PIL import Image
from utils import get_image_caption_from_ml
import os

# from neuraltalk import FeatureExtractor
from gpt3 import generate_caption
from utils import CAPTION_CATEGORIES, check_image

AWS_ACCESS_KEY_ID = os.getenv("AMAZON_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AMAZON_ACCESS_KEY_SECRET")

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:////tmp/test.db"
db = SQLAlchemy(app)
CORS(app, origins="*")


class AIImageCaption(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    caption = db.Column(db.Text)
    category = db.Column(db.Text)
    gen_caption = db.Column(db.Text)


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        try:
            file_extension = request.form.get("image").rsplit(".", 1)[-1]
            current_date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{current_date}.{file_extension}"
            s3_client = boto3.client(
                "s3",
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            )

            presigned_url = s3_client.generate_presigned_url(
                ClientMethod="put_object",
                Params={
                    "Bucket": filename,
                    "Key": filename,
                    "ContentType": f"image/{file_extension}",
                },
                ExpiresIn=3600,  # URL expiration time in seconds (e.g., 1 hour)
            )

            return jsonify({"presigned_url": presigned_url, "filename": filename})
        except Exception as e:
            return jsonify({"error": str(e)}), 500


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

            image = request.form.get("image")

            image_caption_from_image = get_image_caption_from_ml(image)

            if not image or not image_caption_from_image:
                return jsonify({"err": "Couldn't infer anything from the image"}), 500

            ai_generated_captions = generate_caption(image_caption_from_image, category)

            ai_caption = AIImageCaption(
                category=category,
                caption=image_caption_from_image,
                gen_caption=ai_generated_captions,
            )
            db.session.add(ai_caption)
            db.session.commit()
            captions_array = ai_generated_captions.split(",")
            response = [
                {"caption": caption, type: category} for caption in captions_array
            ]

            return jsonify(response)

        except Exception as e:
            print(e)
            return jsonify({"err": str(e)}), 500

    return jsonify({"err": "METHOD is not supported"}), 405


if __name__ == "__main__":
    app.run(debug=True)
