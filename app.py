from flask import Flask, request
from flask import jsonify
# from caption import load_model,get_final_output
import filetype
from neuraltalk import FeatureExtractor
from caption import get_model,get_captions
from flask_sqlalchemy import SQLAlchemy
from PIL import Image
import imagehash
import random 
from gpt3 import generate_caption,generate_prompt
from utils import CAPTION_CATEGORIES

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/test.db'
db = SQLAlchemy(app)

class ImageCaption(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    hash = db.Column(db.Text)
    captions = db.Column(db.Text)
class AIImageCaption(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    caption = db.Column(db.Text)
    category = db.Column(db.Text)
    gen_caption = db.Column(db.Text)

db.create_all()

MAX_LENGTH = 5

feature_extractor = FeatureExtractor()
model = get_model()

def check_image(request):
    f = request.files.get('image')
    if f and filetype.is_image(f):
        return True 
    return False

@app.route("/",methods=['GET', 'POST'])
def home():
    if request.method=="POST":
        category = random.choice(CAPTION_CATEGORIES).lower()

        try :
            is_valid = check_image(request)
            if not is_valid:
                return jsonify({"err": "bad request"
                }),400
            
            image= request.files.get("image")
            hash = str(imagehash.average_hash(Image.open(image)))
            image_caption= ImageCaption.query.filter_by(hash=hash).first()
            
            captions = []
            if image_caption:
                captions = image_caption.captions
                captions = captions.split("-")
            else: 
                feature = feature_extractor(image)
                captions = get_captions(model,feature)
                new_caption = ImageCaption(hash=hash,captions="-".join(caption for caption in captions))
                db.session.add(new_caption)
                db.session.commit()

            if len(captions)==0:
                return jsonify({"err": "couldn't infer anything from the image"
                }),500

            random_caption = random.choice(captions)
            image_caption= AIImageCaption.query.filter_by(caption=random_caption,category=category).first()
            if image_caption:
                ai_generated_captions = image_caption.gen_caption.split("//")
                return jsonify({"caption":ai_generated_captions})

            prompt = generate_prompt(random_caption,category)
            ai_generated_caption = generate_caption(prompt)
            ai_generated_caption = [caption.text.strip() for caption in ai_generated_caption.choices]
            parsed_ai_generated_caption = "//".join(ai_generated_caption)
            
            ai_captions = AIImageCaption(category=category,caption=random_caption,gen_caption=parsed_ai_generated_caption)
            db.session.add(ai_captions)
            db.session.commit()

            return jsonify({"caption": ai_generated_caption})


        except Exception as e:
            print(e)
            return jsonify({"err":str(e)}),500

    return jsonify({
        "err":"METHOD is not supported"
    }) , 405

if __name__=="__main__":
    app.run(debug=True)