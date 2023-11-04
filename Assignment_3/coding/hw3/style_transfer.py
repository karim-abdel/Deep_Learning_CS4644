import numpy as np
import PIL
import torch
import torchvision
from torch.autograd import Variable

from image_utils import preprocess
from style_modules import ContentLoss, StyleLoss, TotalVariationLoss
from style_utils import extract_features, features_from_img, rel_error, style_transfer

dtype = torch.FloatTensor
# Uncomment out the following line if you're on a machine with a GPU set up for PyTorch!
# dtype = torch.cuda.FloatTensor

cnn = torchvision.models.squeezenet1_1(pretrained=True).features
cnn.type(dtype)

# Fix the weights of the pretrained network
for param in cnn.parameters():
    param.requires_grad = False


content_loss = ContentLoss()

style_loss = StyleLoss()

tv_loss = TotalVariationLoss()


# Generate Images
# Composition VII + Tubingen
params1 = {
    "name": "composition_vii_tubingen",
    "content_image": "styles_images/tubingen.jpg",
    "style_image": "styles_images/composition_vii.jpg",
    "image_size": 192,
    "style_size": 512,
    "content_layer": 3,
    "content_weight": 5e-2,
    "style_layers": (1, 4, 6, 7),
    "style_weights": (20000, 500, 12, 1),
    "tv_weight": 5e-2,
    "content_loss": content_loss,
    "style_loss": style_loss,
    "tv_loss": tv_loss,
    "cnn": cnn,
    "dtype": dtype,
}

# Scream + Tubingen
params2 = {
    "name": "scream_tubingen",
    "content_image": "styles_images/tubingen.jpg",
    "style_image": "styles_images/the_scream.jpg",
    "image_size": 192,
    "style_size": 224,
    "content_layer": 3,
    "content_weight": 3e-2,
    "style_layers": [1, 4, 6, 7],
    "style_weights": [200000, 800, 12, 1],
    "tv_weight": 2e-2,
    "content_loss": content_loss,
    "style_loss": style_loss,
    "tv_loss": tv_loss,
    "cnn": cnn,
    "dtype": dtype,
}

# Starry Night + Tubingen
params3 = {
    "name": "starry_tubingen",
    "content_image": "styles_images/tubingen.jpg",
    "style_image": "styles_images/starry_night.jpg",
    "image_size": 192,
    "style_size": 192,
    "content_layer": 3,
    "content_weight": 6e-2,
    "style_layers": [1, 4, 6, 7],
    "style_weights": [300000, 1000, 15, 3],
    "tv_weight": 2e-2,
    "content_loss": content_loss,
    "style_loss": style_loss,
    "tv_loss": tv_loss,
    "cnn": cnn,
    "dtype": dtype,
}


def composition_vii():
    return style_transfer(**params1)


def scream():
    return style_transfer(**params2)


def starry_night():
    return style_transfer(**params3)
