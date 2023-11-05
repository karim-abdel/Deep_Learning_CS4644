import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from captum.attr import (
    GuidedBackprop,
    GuidedGradCam,
    LayerActivation,
    LayerConductance,
    LayerGradCam,
)
from PIL import Image

from captum_utils import *
from data_utils import *
from image_utils import *
from visualizers import GradCam

plt.rcParams["figure.figsize"] = (10.0, 8.0)  # set default size of plots
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"

X, y, class_names = load_imagenet_val(num=5)

gc = GradCam()

# FOR THIS SECTION ONLY, we need to use gradients. We introduce a new model we will use explicitly for GradCAM for this.
def guided_backprop():
    gc_model = torchvision.models.squeezenet1_1(pretrained=True)

    X_tensor = torch.cat(
        [preprocess(Image.fromarray(x)) for x in X], dim=0
    ).requires_grad_(True)
    y_tensor = torch.LongTensor(y)

    # Guided Back-Propagation
    gbp_result = gc.guided_backprop(X_tensor, y_tensor, gc_model)

    plt.figure(figsize=(24, 6.5))
    plt.suptitle("Guided Backprop", fontsize=20)
    for i in range(gbp_result.shape[0]):
        plt.subplot(1, 5, i + 1)
        img = gbp_result[i]
        img = rescale(img)
        plt.imshow(img)
        plt.title(class_names[y[i]])
        plt.axis("off")
    plt.gcf().tight_layout()
    return plt


# GradCam
# GradCAM. We have given you which module(=layer) that we need to capture gradients from, which you can see in conv_module variable below
gc_model = torchvision.models.squeezenet1_1(pretrained=True)
for param in gc_model.parameters():
    param.requires_grad = True


def gradcam():
    X_tensor = torch.cat(
        [preprocess(Image.fromarray(x)) for x in X], dim=0
    ).requires_grad_(True)
    y_tensor = torch.LongTensor(y)
    gradcam_result = gc.grad_cam(X_tensor, y_tensor, gc_model)

    plt.figure(figsize=(24, 6.5))
    plt.suptitle("GradCam", fontsize=20)
    for i in range(gradcam_result.shape[0]):
        gradcam_val = gradcam_result[i]
        img = X[i] + (matplotlib.cm.jet(gradcam_val)[:, :, :3] * 255)
        img = img / np.max(img)
        plt.subplot(1, 5, i + 1)
        plt.imshow(img)
        plt.title(class_names[y[i]])
        plt.axis("off")
    plt.gcf().tight_layout()
    return plt


# As a final step, we can combine GradCam and Guided Backprop to get Guided GradCam.
def guided_gradcam():
    X_tensor = torch.cat(
        [preprocess(Image.fromarray(x)) for x in X], dim=0
    ).requires_grad_(True)
    y_tensor = torch.LongTensor(y)
    gradcam_result = gc.grad_cam(X_tensor, y_tensor, gc_model)
    gbp_result = gc.guided_backprop(X_tensor, y_tensor, gc_model)

    plt.figure(figsize=(24, 6.5))
    plt.suptitle("Guided GradCam", fontsize=20)
    for i in range(gradcam_result.shape[0]):
        gbp_val = gbp_result[i]
        gradcam_val = np.expand_dims(gradcam_result[i], axis=2)

        # Pointwise multiplication and normalization of the gradcam and guided backprop results (2 lines)
        img = gradcam_val * gbp_val

        img = np.expand_dims(img.transpose(2, 0, 1), axis=0)
        img = np.float32(img)
        img = torch.from_numpy(img)
        img = deprocess(img)
        plt.subplot(1, 5, i + 1)
        plt.imshow(img)
        plt.title(class_names[y[i]])
        plt.axis("off")
    plt.gcf().tight_layout()
    return plt


# **************************************************************************************** #
# Captum
model = torchvision.models.squeezenet1_1(pretrained=True)

# We don't want to train the model, so tell PyTorch not to compute gradients
# with respect to model parameters.
for param in model.parameters():
    param.requires_grad = False

# Convert X and y from numpy arrays to Torch Tensors
X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0)
y_tensor = torch.LongTensor(y)


def guided_gradcam_captum():
    conv_module = model.features[12]

    ##############################################################################
    # TODO: Compute/Visualize GuidedBackprop and Guided GradCAM as well.         #
    #       visualize_attr_maps function from captum_utils.py is useful for      #
    #       visualizing captum outputs                                           #
    #       Use conv_module as the convolution layer for gradcam                 #
    ##############################################################################

    gu_backprop = GuidedBackprop(model)

    guid_gc = GuidedGradCam(model, conv_module)

    attr = gu_backprop.attribute(X_tensor, target=y_tensor)

    guided_gc_attr = guid_gc.attribute(X_tensor, target=y_tensor)

    plt = visualize_attr_maps("Guided Backprop and Guided GradCAM Captum", X, y, class_names, [attr, guided_gc_attr], ["Guided Backprop", "Guided GradCAM"])
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return plt


def layer_vis_captum():
    # Try out different layers and see observe how the attributions change
    layer = model.features[3]

    # Example visualization for using layer visualizations
    # layer_act = LayerActivation(model, layer)
    # layer_act_attr = compute_attributions(layer_act, X_tensor)
    # layer_act_attr_sum = layer_act_attr.mean(axis=1, keepdim=True)

    ##############################################################################
    # TODO: Visualize Individual Layer Gradcam and Layer Conductance (similar    #
    # to what we did for the other captum sections, using our helper methods),   #
    # but with some preprocessing calculations.                                  #
    #                                                                            #
    # You can refer to the LayerActivation example above and you should be       #
    # using 'layer' given above for this section                                 #
    #                                                                            #
    # Also note that, you would need to customize your 'attr_preprocess'         #
    # parameter that you send along to 'visualize_attr_maps' as the default      #
    # 'attr_preprocess' is written to only to handle multi channel attributions. #
    #                                                                            #
    # For layer gradcam look at the usage of the parameter relu_attributions     #
    ##############################################################################
    lay_gc = LayerGradCam(model, layer)

    lay_con = LayerConductance(model, layer)

    attr = lay_gc.attribute(X_tensor, target=y_tensor, relu_attributions=True)

    cd = lay_con.attribute(X_tensor, target=y_tensor)
    
    cd_sum = cd.mean(axis=1, keepdim=True)

    plt = visualize_attr_maps(
        "Layer GradCAM and Layer Conductance Captum", X, y, class_names,
        [attr, cd_sum], ["Layer GradCAM", "Layer Conductance"], lambda attr: attr.detach().numpy().squeeze()
    )

    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return plt
