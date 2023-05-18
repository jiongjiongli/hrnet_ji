import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
import onnx
import onnxsim

from nets.hrnet import HRnet


class HRnetPredict(HRnet):
    def __init__(self, num_classes = 21, backbone = 'hrnetv2_w18', pretrained = False):
        super(HRnetPredict, self).__init__(num_classes=num_classes, backbone=backbone, pretrained=pretrained)

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        x = self.backbone(inputs)

        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True)

        x = torch.cat([x[0], x1, x2, x3], 1)

        x = self.last_layer(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)

        # Shape: [B, C=1, H, W]
        x = torch.argmax(x, dim=1, keepdim=True)
        return x


class HRnetConverter:
    def __init__(self, model_path=None, num_classes = 21, backbone = 'hrnetv2_w18', pretrained = False, input_shape=None):
        self.input_shape = input_shape
        self.net = HRnetPredict(num_classes=num_classes, backbone=backbone, pretrained=pretrained)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))

    def convert_to_onnx(self, simplify, onnx_save_path):
        im = torch.zeros(1, 3, *self.input_shape).to('cpu')  # image size(1, 3, 512, 512) BCHW
        input_layer_names = ["images"]
        output_layer_names = ["output"]

        # Export the model
        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(self.net,
                          im,
                          f=onnx_save_path,
                          verbose=False,
                          opset_version=12,
                          training=torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=True,
                          input_names=input_layer_names,
                          output_names=output_layer_names,
                          dynamic_axes=None)

        # Checks
        model_onnx = onnx.load(onnx_save_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify onnx
        if simplify:
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        print('Onnx model save as {}'.format(model_path))


def main():
    simplify=True
    onnx_save_path = "/project/train/models/seg_model.onnx"
    model_path = "/project/train/models/best_epoch_weights.pth"
    hrnet = HRnetConverter(model_path=model_path,
                           num_classes=2,
                           backbone="hrnetv2_w32",
                           input_shape=[512,512])
    hrnet.convert_to_onnx(simplify, onnx_save_path)


if __name__ == '__main__':
    main()
