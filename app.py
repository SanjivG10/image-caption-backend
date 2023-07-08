import datetime
import os
import bcrypt
import jwt

import boto3
import botocore
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from flask_sqlalchemy import SQLAlchemy

# from neuraltalk import FeatureExtractor
from gpt3 import generate_caption
from utils import CAPTION_CATEGORIES, get_image_caption_from_ml

AWS_ACCESS_KEY_ID = os.getenv("AMAZON_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AMAZON_ACCESS_KEY_SECRET")
debug = os.getenv("debug")
if debug:
    debug = True

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:////tmp/user.db"
app.config["SECRET_KEY"] = "HarryMaguire"  # Set your own secret key
CORS(app, origins="*")


db = SQLAlchemy(app)


class AIImageCaption(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    caption = db.Column(db.Text)
    url = db.Column(db.Text)
    category = db.Column(db.Text)
    gen_caption = db.Column(db.Text)
    user = db.Column(db.Text)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

    def __init__(self, email, password):
        self.email = email
        self.password = password

    def check_password(self, password):
        return bcrypt.checkpw(password.encode("utf-8"), self.password)

    def generate_token(self):
        payload = {
            "email": self.email,
            "exp": datetime.datetime.utcnow() + datetime.timedelta(days=1000),
        }
        return jwt.encode(payload, app.config["SECRET_KEY"], algorithm="HS256")


with app.app_context():
    db.create_all()


def get_caption_by_url(url):
    try:
        document = AIImageCaption.query.filter_by(url=url).first()
        return document.caption
    except Exception as e:
        return None


@app.route("/register", methods=["POST"])
def register():
    try:
        data = request.get_json()
        email = data.get("email")
        password = data.get("password")

        if not email or not password:
            return jsonify({"err": "Email and password are required fields."}), 400

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return (
                jsonify(
                    {"err": "Email already exists. Please choose a different email."}
                ),
                409,
            )

        hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

        user = User(email=email, password=hashed_password)
        db.session.add(user)
        db.session.commit()

        token = user.generate_token()

        return jsonify({"token": token})

    except Exception as e:
        return jsonify({"err": str(e)}), 500


@app.route("/login", methods=["POST"])
def login():
    try:
        data = request.get_json()
        email = data.get("email")
        password = data.get("password")

        if not email or not password:
            return jsonify({"err": "Email and password are required fields."}), 400

        user = User.query.filter_by(email=email).first()
        if not user or not user.check_password(password):
            return jsonify({"err": "Invalid email or password."}), 401

        token = user.generate_token()

        return jsonify({"token": token})

    except Exception as e:
        return jsonify({"err": str(e)}), 500


def authenticate_request():
    token = request.headers.get("Authorization")
    if not token:
        return jsonify({"err": "Authorization token is missing."}), 401

    try:
        payload = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
        email = payload["email"]
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"err": "Invalid token."}), 401

        return user

    except jwt.ExpiredSignatureError:
        return jsonify({"err": "Token has expired."}), 401

    except jwt.InvalidTokenError:
        return jsonify({"err": "Invalid token."}), 401

    except Exception as e:
        print(str(e))
        return jsonify({"err": str(e)}), 401


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        user_or_err = authenticate_request()
        if not isinstance(user_or_err, User):
            return user_or_err
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
        user_or_err = authenticate_request()
        if not isinstance(user_or_err, User):
            return user_or_err
        email = user_or_err.email

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
                user=email,
            )
            db.session.add(ai_caption)
            db.session.commit()
            captions_array = ai_generated_captions.split("\n")
            response = [
                {"caption": caption.strip(), "type": category}
                for caption in captions_array
            ]

            return jsonify(response)

        except Exception as e:
            print(e)
            return jsonify({"err": str(e)}), 500

    return jsonify({"err": "METHOD is not supported"}), 405


if debug:
    if __name__ == "__main__":
        app.run(debug=True)
