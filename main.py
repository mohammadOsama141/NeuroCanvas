from PIL import Image
import requests
from transformers import SamModel, SamProcessor
import torch
import cv2
import numpy as np
from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from segment import Segmenter


app = FastAPI()

torch.set_grad_enabled(False)

device = 'cuda' if torch.cuda.is_available() else "cpu"

combinedMask = None

app.mount("/static", StaticFiles(directory="static"), name="static")


raw_image = Image.open('static/img.png').convert("RGB")


segmenter = Segmenter()

@app.get('/')
def index():
    return FileResponse('templates/index.html', headers={"file": 'img.png'})



@app.get('/add_segment')
async def add_segment(request: Request):
    global combinedMask

    x = request.query_params.get('x')
    y = request.query_params.get('y')
    dilation = request.query_params.get('dilation')

    mask = segmenter.get_mask(raw_image, [[[x, y]]], int(dilation))

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

    return JSONResponse(content={'image_path': '/static/result.png'})



@app.get('/remove_segment')
async def remove_segment(request: Request):
    global combinedMask

    x = request.query_params.get('x')
    y = request.query_params.get('y')

    mask_to_remove = segmenter.get_mask(raw_image, [[[x, y]]])

    combinedMask = np.logical_and(combinedMask, np.logical_not(mask_to_remove))

    orig_img = np.array(raw_image)
    combinedMask_display = np.array(combinedMask, dtype=np.uint8) * 1
    combinedMask_display = combinedMask_display.reshape(combinedMask_display.shape[0], -1, 1)
    mask_img = np.concatenate((combinedMask_display * 255, combinedMask_display * 0, combinedMask_display * 0), 2)
    final_img = cv2.addWeighted(orig_img, 1, mask_img, 0.5, 0)
    final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('static/result.png', final_img)

    return JSONResponse(content={'image_path': '/static/result.png'})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
