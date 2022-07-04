from flask import Flask, request
from flask import jsonify
# from caption import load_model,get_final_output
import filetype


app = Flask(__name__)

MAX_LENGTH = 5


def check_image(request):
    f = request.files.get('image')
    if f and filetype.is_image(f):
        return True 
    return False

@app.route("/",methods=['GET', 'POST'])
def home():
    if request.method=="POST":
        is_valid = check_image(request)
        if not is_valid:
            return jsonify({"err": "bad request"
            }),400

        f= request.files.get("image")
        image_path = [f]
        output = []

        if len(output)==0:
            return jsonify({"err": "couldn't infer anything from the image"
            }),500

        return jsonify(output)

    return jsonify({
        "err":"METHOD is not supported"
    }) , 405

if __name__=="__main__":
    app.run(debug=True)