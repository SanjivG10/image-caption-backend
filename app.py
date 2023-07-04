import datetime
import random

import boto3
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from flask_sqlalchemy import SQLAlchemy
from utils import get_image_caption_from_ml
import os
import botocore

# from neuraltalk import FeatureExtractor
from gpt3 import generate_caption
from utils import CAPTION_CATEGORIES

AWS_ACCESS_KEY_ID = os.getenv("AMAZON_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AMAZON_ACCESS_KEY_SECRET")

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:////tmp/test.db"
CORS(app, origins="*")


db = SQLAlchemy(app)


class AIImageCaption(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    caption = db.Column(db.Text)
    url = db.Column(db.Text)
    category = db.Column(db.Text)
    gen_caption = db.Column(db.Text)


with app.app_context():
    db.create_all()


def get_caption_by_url(url):
    try:
        document = AIImageCaption.query.filter_by(url=url).first()
        return document.caption
    except Exception as e:
        return None


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
                region_name="us-west-2",
                config=botocore.client.Config(signature_version="s3v4"),
            )
            bucketname = "imagecaptionai"

            presigned_url = s3_client.generate_presigned_url(
                ClientMethod="put_object",
                Params={
                    "Bucket": bucketname,
                    "Key": filename,
                    "ContentType": f"image/{file_extension}",
                },
                ExpiresIn=3600,
            )

            return jsonify({"presigned_url": presigned_url, "filename": filename})
        except Exception as e:
            return jsonify({"error": str(e)}), 500


@cross_origin(origins="*")
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            category = request.form.get("category", CAPTION_CATEGORIES[0]).lower()
            if not category in CAPTION_CATEGORIES:
                return jsonify({"err": "Chosen category doesn't exist."}), 400

            image_url = request.form.get("image")
            image_caption_from_image = get_caption_by_url(image_url)
            if not image_caption_from_image:
                print("no image with same url found, generating caption using ml")
                image_caption_from_image = get_image_caption_from_ml(image_url)
                image_caption_from_image = str(image_caption_from_image[0])
                print("Caption generated =>", image_caption_from_image)

            if not image_url or not image_caption_from_image:
                return jsonify({"err": "Couldn't infer anything from the image"}), 500

            print("AI image caption generating")
            ai_generated_captions = generate_caption(image_caption_from_image, category)
            print("AI image caption generated")

            ai_caption = AIImageCaption(
                category=category,
                caption=image_caption_from_image,
                gen_caption=ai_generated_captions,
                url=image_url,
            )
            db.session.add(ai_caption)
            db.session.commit()
            captions_array = ai_generated_captions.split(",")
            response = [
                {"caption": caption.strip(), "type": category}
                for caption in captions_array
            ]

            return jsonify(response)

        except Exception as e:
            print(e)
            return jsonify({"err": str(e)}), 500

    return jsonify({"err": "METHOD is not supported"}), 405


if __name__ == "__main__":
    app.run(debug=True)
