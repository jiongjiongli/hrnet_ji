import json
import onnxruntime
from PIL import Image
import numpy as np
import cv2
import time
import torch.nn.functional as F
import torch
def init():
    # h5_to_pb()
    session = onnxruntime.InferenceSession("/project/train/models/seg_model.onnx",providers=['CUDAExecutionProvider'])
    return session

colors = [(0, 0, 0), (0, 0, 128)]
def process_image(handle=None, input_image=None, args=None, **kwargs):
    model = handle
    vis = True
    img_size= [512,512]

    h, w, _ = input_image.shape
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    
    
    img = cv2.resize(input_image, img_size, cv2.INTER_LINEAR)
    img = np.array(img, np.float32)
    img -= np.array([123.675, 116.28, 103.53], np.float32)
    img /= np.array([58.395, 57.12, 57.375], np.float32)
    img = np.expand_dims(np.transpose(np.array(img), (2, 0, 1)), 0)
    t0 = time.time()
    y_pre = model.run([], {'images': img})[0][0]
    t1 = time.time()

    y_pre = np.transpose(np.array(y_pre, np.float32), (1, 2, 0))
    y_pre = torch.from_numpy(y_pre)
    y_pre = y_pre.cpu()
    y_pre = F.softmax(y_pre, dim=-1).cpu().numpy()

    y_pre = np.argmax(y_pre, axis=-1)

    y_pre = np.squeeze(y_pre).astype(np.uint8)

    y_pre = cv2.resize(y_pre, (w, h), interpolation=cv2.INTER_NEAREST)
    args = json.loads(args)
    mask_output_path = args['mask_output_path']    
    cv2.imwrite(mask_output_path, y_pre)
    if vis:
        seg_img = np.reshape(np.array(colors, np.uint8)[np.reshape(y_pre, [-1])], [h, w, -1])
        render_map = seg_img * 0.8 + cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(mask_output_path.replace('.png','.jpg'),render_map)
    return json.dumps({'mask': mask_output_path}, indent=4)


if __name__ == '__main__':

    input_image = cv2.imread("1.jpg")
    for i in range(1):
        result = process_image(init(), input_image,
                           '{"mask_output_path": "onnx.png"}')