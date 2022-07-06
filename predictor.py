from model.densecap import densecap_resnet50_fpn
import torch
from PIL import Image 
import torchvision.transforms as transforms


def img_to_tensor(img_paths):

    img_tensors = []
    for img_path in img_paths:
        img = Image.open(img_path).convert("RGB")
        img_tensors.append(transforms.ToTensor()(img))

    return img_tensors

def describe_image(model, img_paths, device):

    all_results = []

    with torch.no_grad():

        model.to(device)
        model.eval()
        image_tensors= img_to_tensor(img_paths)
        input_ = [t.to(device) for t in image_tensors]

        results = model(input_)

        all_results.extend([{k:v.cpu() for k,v in r.items()} for r in results])

    return all_results

def load_model():
    model_args = {
        "extract":False,
        "box_per_img":10,
        "model_checkpoint": "train_file.tar",
    "backbone_pretrained": True,
    "return_features": False,
    "feat_size": 4096,
    "hidden_size": 512,
    "max_len": 16,
    "emb_size": 512,
    "rnn_num_layers": 1,
    "vocab_size": 10629,
    "fusion_type": "init_inject",
    "detect_loss_weight": 1.0,
    "caption_loss_weight": 1.0,
    "lr": 0.0001,
    "caption_lr": 0.001,
    "weight_decay": 0.0,
    "batch_size": 4,
    "use_pretrain_fasterrcnn": True,
    "box_detections_per_img": 50
        }

    model = densecap_resnet50_fpn(backbone_pretrained=model_args['backbone_pretrained'],
                                  return_features=model_args["extract"],
                                  feat_size=model_args['feat_size'],
                                  hidden_size=model_args['hidden_size'],
                                  max_len=model_args['max_len'],
                                  emb_size=model_args['emb_size'],
                                  rnn_num_layers=model_args['rnn_num_layers'],
                                  vocab_size=model_args['vocab_size'],
                                  fusion_type=model_args['fusion_type'],
                                  box_detections_per_img=model_args["box_per_img"])

    checkpoint = torch.load(model_args["model_checkpoint"])
    model.load_state_dict(checkpoint['model'])

    return model