import json
import onnxruntime
from PIL import Image
import numpy as np
import cv2
import time
import torch.nn.functional as F
import torch
from pathlib import Path

import sys
sys.path.append('/project/train/src_repo/hrnet_ji')
from hrnet import HRnet_Segmentation


def init():
    # h5_to_pb()
    # session = onnxruntime.InferenceSession("/project/train/models/seg_model.onnx",providers=['CUDAExecutionProvider'])
    # return session
    hrnet = HRnet_Segmentation(model_path ="/project/train/models/best_epoch_weights.pth",
                               num_classes = 2,
                               backbone = "hrnetv2_w32",
                               input_shape=[512,512],
                               mix_type=3)
    return hrnet

colors = [(0, 0, 0), (0, 0, 128)]
def process_image(handle=None, input_image=None, args=None, **kwargs):
    model = handle
    vis = False
    img_size= [512,512]

    h, w, _ = input_image.shape
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    img = cv2.resize(input_image, img_size, cv2.INTER_LINEAR)
    img = np.array(img, np.float32)
    img -= np.array([123.675, 116.28, 103.53], np.float32)
    img /= np.array([58.395, 57.12, 57.375], np.float32)
    img = np.expand_dims(np.transpose(np.array(img), (2, 0, 1)), 0)
    # t0 = time.time()
    # y_pre = model.run([], {'images': img})[0][0]
    # t1 = time.time()

    with torch.no_grad():
        images = torch.from_numpy(img).cuda()

        y_pre = model.net(images)[0]
        y_pre = F.softmax(y_pre.permute(1, 2, 0), dim=-1).cpu().numpy()

    # y_pre = np.transpose(np.array(y_pre, np.float32), (1, 2, 0))
    # y_pre = torch.from_numpy(y_pre)
    # y_pre = y_pre.cpu()
    # y_pre = F.softmax(y_pre, dim=-1).cpu().numpy()

    y_pre = np.argmax(y_pre, axis=-1)

    y_pre = y_pre.astype(np.uint8)

    # y_pre = np.squeeze(y_pre).astype(np.uint8)

    y_pre = cv2.resize(y_pre, (w, h), interpolation=cv2.INTER_NEAREST)

    # input_image = Image.fromarray(input_image)
    # y_pre = handle.detect_image(input_image)

    args = json.loads(args)
    mask_output_path = args['mask_output_path']
    cv2.imwrite(mask_output_path, y_pre)
    # if vis:
    #     seg_img = np.reshape(np.array(colors, np.uint8)[np.reshape(y_pre, [-1])], [h, w, -1])
    #     render_map = seg_img * 0.8 + cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    #     cv2.imwrite(mask_output_path.replace('.png','.jpg'),render_map)
    return json.dumps({'model_data': {'mask': mask_output_path}}, indent=4)


if __name__ == '__main__':
    model = init()
    data_dir_path = r'/home/data'
    data_dir_path = Path(data_dir_path)
    data_file_paths = list(data_dir_path.rglob('*.jpg'))
    print('total file count:', len(data_file_paths))
    data_file_paths = data_file_paths[:20]
    print('test file count:', len(data_file_paths))
    process_image_durations = []
    durations = []

    for data_file_path in data_file_paths:
        input_image = cv2.imread(data_file_path.as_posix())
        start_time = time.time()
        result = process_image(model, input_image,
                               '{"mask_output_path": "onnx.png"}')

        end_time = time.time()
        process_image_duration = end_time - start_time
        process_image_durations.append(process_image_duration)

        end_time = time.time()
        duration = end_time - start_time
        durations.append(duration)
        print('duration:', duration)
        print('process_image_duration:', process_image_duration)

    duration_sum = sum(durations)
    print('duration_average:', duration_sum / len(durations))
    print('duration_sum:', duration_sum)
    print('process_image_duration_average:', sum(process_image_durations) / len(process_image_durations))
    print('process_image_duration_sum:', sum(process_image_durations))

