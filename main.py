from PIL import Image
import requests
from transformers import SamModel, SamProcessor
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, jsonify, request, render_template
import os

app = Flask(__name__)

torch.set_grad_enabled(False)

device = 'cuda' if torch.cuda.is_available() else "cpu"
# print("\n", device, "\n" )

model = SamModel.from_pretrained("facebook/sam-vit-base")
model.to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

combinedMask = None


def return_mask(img, input_points, dilation=5):
    inputs = processor(img, input_points=input_points, return_tensors="pt").to(device)
    outputs = model(**inputs)
    masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(),
                                                         inputs["original_sizes"].cpu(),
                                                         inputs["reshaped_input_sizes"].cpu())
    mask = np.array(masks[0][0][-1])
    print(dilation, mask.shape, mask.dtype)
    mask = Dilation(mask, dilation)
    return mask


raw_image = Image.open('static/img.png').convert("RGB")


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', file='img.png')


# add mask feature
@app.route('/add_segment', methods=['GET'])
def add_segment():
    global combinedMask
    x = request.args.get('x')
    y = request.args.get('y')
    dilation = request.args.get('dilation')
    mask = return_mask(raw_image, [[[x, y]]], int(dilation))

    if combinedMask is None:
        combinedMask = mask
    else:
        combinedMask = np.logical_or(combinedMask, mask)

    orig_img = np.array(raw_image)
    combinedMask_display = np.array(combinedMask, dtype=np.uint8) * 1
    combinedMask_display = combinedMask_display.reshape(combinedMask_display.shape[0], -1, 1)
    mask_img = np.concatenate((combinedMask_display * 255, combinedMask_display * 0, combinedMask_display * 0), 2)
    final_img = cv2.addWeighted(orig_img, 1, mask_img, 0.5, 0)
    final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('static/result.png', final_img)
    return jsonify({'image_path': '/static/result.png'})


# removing feature
@app.route('/remove_segment', methods=['GET'])
def remove_segment():
    global combinedMask
    x = request.args.get('x')
    y = request.args.get('y')
    mask_to_remove = return_mask(raw_image, [[[x, y]]])

    combinedMask = np.logical_and(combinedMask, np.logical_not(mask_to_remove))  # remove segment from whole combineMask

    orig_img = np.array(raw_image)
    combinedMask_display = np.array(combinedMask, dtype=np.uint8) * 1
    combinedMask_display = combinedMask_display.reshape(combinedMask_display.shape[0], -1, 1)
    mask_img = np.concatenate((combinedMask_display * 255, combinedMask_display * 0, combinedMask_display * 0), 2)
    final_img = cv2.addWeighted(orig_img, 1, mask_img, 0.5, 0)
    final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('static/result.png', final_img)
    return jsonify({'image_path': '/static/result.png'})


def Dilation(mask, dilationVal):
    if mask is not None and mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    kernel = np.ones((dilationVal, dilationVal), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1) if mask is not None else None


if __name__ == '__main__':
    app.run(debug=False)

#
# img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
# raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
# input_points = [[[1700, 850]]] # 2D localization of a window
#
# mask = return_mask(raw_image, input_points)
#
# img = np.array(raw_image)
# plt.imshow(img)
# plt.plot(input_points[0][0][0], input_points[0][0][1], 'ro')
#
# plt.imshow(mask, alpha=mask*0.5)
#
# plt.imshow(img)
# plt.plot(1700, 850, 'ro')
