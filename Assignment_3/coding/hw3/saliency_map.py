import matplotlib.pyplot as plt
import torch
import torchvision
from captum.attr import IntegratedGradients, Saliency
from PIL import Image
from torch.autograd import Variable

from captum_utils import *
from data_utils import *
from image_utils import *
from visualizers import SaliencyMap

plt.rcParams["figure.figsize"] = (10.0, 8.0)  # set default size of plots
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"

# Download and load the pretrained SqueezeNet model.
model = torchvision.models.squeezenet1_1(pretrained=True)

# We don't want to train the model, so tell PyTorch not to compute gradients
# with respect to model parameters.
for param in model.parameters():
    param.requires_grad = False

X, y, class_names = load_imagenet_val(num=5)


def manual_saliency_maps():
    sm = SaliencyMap()
    plt = sm.show_saliency_maps(X, y, class_names, model)
    return plt


def captum_saliency_maps():
    # Convert X and y from numpy arrays to Torch Tensors
    X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0)
    y_tensor = torch.LongTensor(y)

    # Example with captum
    # Computing and visualizing Integrated Gradient

    # int_grads = IntegratedGradients(model)
    # attr_ig = compute_attributions(int_grads, X_tensor, target=y_tensor, n_steps=10)
    # visualize_attr_maps("Grads Captum", X, y, class_names, [attr_ig], ['Integrated Gradients'])

    ##############################################################################
    # TODO: Compute/Visualize Saliency using captum.                             #
    #       visualize_attr_maps function from captum_utils.py is useful for      #
    #       generating plots of captum outputs                                   #
    #       You can refer to the 'Integrated gradients' visualization            #
    #       in the comments above this section as an example                     #
    ##############################################################################

    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################

    return plt
