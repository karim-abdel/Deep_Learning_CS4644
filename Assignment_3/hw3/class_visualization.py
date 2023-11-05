import torch
import torchvision

from data_utils import load_imagenet_val
from visualizers import ClassVisualization

# Download and load the pretrained SqueezeNet model.
model = torchvision.models.squeezenet1_1(pretrained=True)

# We don't want to train the model, so tell PyTorch not to compute gradients
# with respect to model parameters.
for param in model.parameters():
    param.requires_grad = False

dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to use GPU
model.type(dtype)

cv = ClassVisualization()
_, _, class_names = load_imagenet_val(num=5)

targets = [366]


def visualize_class(target_y):
    return cv.create_class_visualization(target_y, class_names, model, dtype)
