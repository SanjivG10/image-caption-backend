from captioning import utils, models
import torch


def get_model():
    infos = utils.misc.pickle_load(open("best.pkl", "rb"))
    infos["opt"].vocab = infos["vocab"]
    model = models.setup(infos["opt"])
    model.load_state_dict(
        torch.load("model-best.pth", map_location=torch.device("cpu"))
    )
    return model


def get_captions(model, img_feature):
    return model.decode_sequence(
        model(
            img_feature.mean(0)[None],
            img_feature[None],
            mode="sample",
            opt={"beam_size": 5, "sample_method": "beam_search", "sample_n": 5},
        )[0]
    )
