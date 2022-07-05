from PIL import Image

CAPTION_CATEGORIES = ["happy","inspiring","sad","motivation","sarcastic","mind-blowing","real life"]

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