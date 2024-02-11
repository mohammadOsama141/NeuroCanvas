import torch
import numpy as np
from transformers import SamModel, SamProcessor
import cv2


def dilation(mask, dilation_radius):
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    kernel = np.ones((dilation_radius, dilation_radius), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1) if mask is not None else None


class Segmenter:
    def __init__(self, model_name='facebook/sam-vit-base'):
        self.model = SamModel.from_pretrained(model_name)
        self.processor = SamProcessor.from_pretrained(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def get_mask(self, img, input_points, dilation_radius=5):
        inputs = self.processor(img, input_points=input_points, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        masks = self.processor.image_processor.post_process_masks(outputs.pred_masks.cpu(),
                                                                  inputs["original_sizes"].cpu(),
                                                                  inputs["reshaped_input_sizes"].cpu())
        mask = np.array(masks[0][0][-1])
        mask = dilation(mask, dilation_radius)
        return mask
