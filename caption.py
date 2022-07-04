from captioning import utils,models
import torch
from neuraltalk import FeatureExtractor
from PIL import Image 

feature_extractor = FeatureExtractor()


infos = utils.misc.pickle_load(open('best.pkl', 'rb'))
infos['opt'].vocab = infos['vocab']
model = models.setup(infos['opt'])
model.load_state_dict(torch.load('model-best.pth'))


def get_captions(img_feature):
    return model.decode_sequence(model(img_feature.mean(0)[None], img_feature[None], mode='sample', opt={'beam_size':5, 'sample_method':'beam_search', 'sample_n':5})[0])

feature = feature_extractor("image.PNG")
captions = get_captions(feature)
print(captions)