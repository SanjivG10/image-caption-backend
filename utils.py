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


def parse_box_cap_scores(results,idx_to_token):
    res = []
    for box, cap, score in zip(results['boxes'], results['caps'], results['scores']):
            r = {
                'box': [round(c, 2) for c in box.tolist()],
                'score': round(score.item(), 2),
                'cap': ' '.join(idx_to_token[idx] for idx in cap.tolist()
                                if idx_to_token[idx] not in ['<pad>', '<bos>', '<eos>'])
            }
            res.append(r)
    return res

def get_caption_from_res(response):
    captions = []
    for res in response:
        captions.append(res['cap'])
    return captions