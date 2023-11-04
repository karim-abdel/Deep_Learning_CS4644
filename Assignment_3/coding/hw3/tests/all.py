import unittest

import numpy as np
import PIL
import torch
import torchvision
from torch.autograd import Variable

from image_utils import preprocess
from style_modules import ContentLoss, StyleLoss, TotalVariationLoss
from style_utils import extract_features, features_from_img, rel_error, style_transfer


class Test(unittest.TestCase):
    def setUp(self):
        self.answers = np.load("data/style-transfer-checks.npz")
        self.cnn = torchvision.models.squeezenet1_1(pretrained=True).features

        self.dtype = torch.FloatTensor
        # Uncomment out the following line if you're on a machine with a GPU set up for PyTorch!
        # self.dtype = torch.cuda.FloatTensor

        self.cnn.type(self.dtype)

    def test_content_loss(self):
        content_loss = ContentLoss()
        correct = self.answers["cl_out"]

        content_image = "styles_images/tubingen.jpg"
        image_size = 192
        content_layer = 3
        content_weight = 6e-2

        c_feats, content_img_var = features_from_img(
            content_image, image_size, self.cnn, self.dtype
        )

        bad_img = Variable(torch.zeros(*content_img_var.data.size()))
        feats = extract_features(bad_img, self.cnn)

        student_output = content_loss(
            content_weight, c_feats[content_layer], feats[content_layer]
        ).data.numpy()
        self.assertAlmostEqual(rel_error(correct, student_output), 0, places=3)

    def test_gram_matrix(self):
        style_loss = StyleLoss()
        correct = self.answers["gm_out"]

        style_image = "styles_images/starry_night.jpg"
        style_size = 192
        feats, _ = features_from_img(style_image, style_size, self.cnn, self.dtype)
        student_output = style_loss.gram_matrix(feats[5].clone()).data.numpy()
        self.assertAlmostEqual(rel_error(correct, student_output), 0, places=3)

    def test_style_loss(self):
        style_loss = StyleLoss()
        correct = self.answers["sl_out"]

        content_image = "styles_images/tubingen.jpg"
        style_image = "styles_images/starry_night.jpg"
        image_size = 192
        style_size = 192
        style_layers = [1, 4, 6, 7]
        style_weights = [300000, 1000, 15, 3]

        c_feats, _ = features_from_img(content_image, image_size, self.cnn, self.dtype)
        feats, _ = features_from_img(style_image, style_size, self.cnn, self.dtype)
        style_targets = []
        for idx in style_layers:
            style_targets.append(style_loss.gram_matrix(feats[idx].clone()))

        student_output = style_loss(
            c_feats, style_layers, style_targets, style_weights
        ).data.numpy()
        self.assertAlmostEqual(rel_error(correct, student_output), 0, places=3)

    def test_tv_loss(self):
        tv_loss = TotalVariationLoss()
        correct = self.answers["tv_out"]

        content_image = "styles_images/tubingen.jpg"
        image_size = 192
        tv_weight = 2e-2

        content_img = preprocess(PIL.Image.open(content_image), size=image_size)
        content_img_var = Variable(content_img.type(self.dtype))

        student_output = tv_loss(content_img_var, tv_weight).data.numpy()
        self.assertAlmostEqual(rel_error(correct, student_output), 0, places=3)
