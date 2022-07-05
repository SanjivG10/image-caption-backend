from PIL import Image

CAPTION_CATEGORIES = ["happy","sarcastic","sad","exciting","angry","romantic","nostalgic","motivational"]

def check_image(request):
    f = request.files.get('image')
    try:
        img = Image.open(f)
        w,h = img.size
        if w>50 and h>50:
            return True
        return False 
    except IOError:
        return False