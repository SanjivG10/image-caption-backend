from flask import Flask, request
from flask import jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import pickle
from gpt3 import generate_caption, generate_prompt

from predictor import describe_image, load_model
from utils import CAPTION_CATEGORIES, check_image, parse_box_cap_scores,get_caption_from_res
import torch
import random


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/test.db'
db = SQLAlchemy(app)
CORS(app)
class AIImageCaption(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    caption = db.Column(db.Text)
    category = db.Column(db.Text)
    gen_caption = db.Column(db.Text)


MAX_LENGTH = 5

model = load_model()
with open("checkpoint.pkl","rb") as f:
    look_up_tables = pickle.load(f)
idx_to_token = look_up_tables['idx_to_token']

device = torch.device("cpu")

@app.route("/",methods=['GET', 'POST'])
def home():
    if request.method=="POST":
        try:
            category = random.choice(CAPTION_CATEGORIES).lower()
            category = request.form.get("category",CAPTION_CATEGORIES[0]).lower()
            if not category in CAPTION_CATEGORIES:
                return jsonify({"err":"Chosen category doesn't exist."}),400

            is_valid = check_image(request)
            if not is_valid:
                return jsonify({"err": "bad request"
                }),400

            image= request.files.get("image")
            image_results = describe_image(model,[image],device)
            response = parse_box_cap_scores(image_results[0],idx_to_token)
            captions = get_caption_from_res(response)
            random_caption = random.choice(captions)
            image_caption = AIImageCaption.query.filter_by(caption=random_caption,category=category).first()
            if image_caption:
                ai_generated_captions = image_caption.gen_caption.split("{}")
                return jsonify({"caption":ai_generated_captions[0],"type":category})
            
            prompt = generate_prompt(random_caption,category)  
            ai_generated_caption = generate_caption(prompt)
            ai_generated_caption = [caption.text.strip() for caption in ai_generated_caption.choices]
            parsed_ai_generated_caption = "{}".join(ai_generated_caption)
            ai_captions = AIImageCaption(category=category,caption=random_caption,gen_caption=parsed_ai_generated_caption)
            db.session.add(ai_captions)
            db.session.commit()
            return jsonify({"caption": ai_generated_caption[0],"type":category})

        except Exception as e:
            return jsonify({"err":str(e)}),500

    return jsonify({
        "err": "Invalid method"
    }) ,405


if __name__=="__main__":
    db.create_all()
    app.run(debug=True)