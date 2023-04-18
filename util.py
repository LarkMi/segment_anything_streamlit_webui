
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
import torch
from PIL import Image
import numpy as np
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import random
from distinctipy import distinctipy
import streamlit as st

def get_checkpoint_path(model):
    if model == 'vit_l':
        return 'checkpoint/sam_vit_l_0b3195.pth'
    elif model == 'vit_b':
        return 'checkpoint/sam_vit_b_01ec64.pth'
    elif model == 'vit_h':
        return 'checkpoint/sam_vit_h_4b8939.pth'


@st.cache_data
def get_color():
    return distinctipy.get_colors(200)

@st.cache_resource
def get_model(model):
    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    build_sam      = sam_model_registry[model]
    model          = build_sam(checkpoint=get_checkpoint_path(model)).to(device)
    predictor      = SamPredictor(model)
    mask_generator = SamAutomaticMaskGenerator(model)
    torch.cuda.empty_cache()
    return predictor, mask_generator

@st.cache_data
def show_everything(sorted_anns):
    if len(sorted_anns) == 0:
        return
    #sorted_anns = sorted(anns, key=(lambda x: x['stability_score']), reverse=True)
    h, w        = sorted_anns[0]['segmentation'].shape[-2:]
    #sorted_anns = sorted_anns[:int(len(sorted_anns) * stability_score/100)]
    if sorted_anns == []:
        return np.zeros((h,w,4)).astype(np.uint8)
    mask = np.zeros((h,w,4))
    for ann in sorted_anns:
        m      = ann['segmentation']
        color  = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        mask  += m.reshape(h,w,1) * color.reshape(1, 1, -1)
    mask = mask * 255
    st.success('Process completed！', icon="✅")
    return mask.astype(np.uint8)

@st.cache_data
def show_click(input_masks_color):
    h, w         = input_masks_color[0][0].shape[-2:]
    masks_total = np.zeros((h,w,4)).astype(np.uint8)
    for mask, color in input_masks_color:
        if np.array_equal(mask,np.array([])):continue
        masks = np.zeros((h,w,4)).astype(np.uint8)
        masks = masks + mask.reshape(h,w,1).astype(np.uint8)
        masks = masks.astype(bool).astype(np.uint8)
        masks = masks * 255 * color.reshape(1, 1, -1)
        masks_total += masks.astype(np.uint8)
    st.success('Process completed!', icon="✅")
    return masks_total

@st.cache_data
def model_predict_everything(im,model):
    predictor, mask_generator = get_model(model)
    torch.cuda.empty_cache()
    return mask_generator.generate(im)

@st.cache_data
def model_predict_click(im,input_points,input_labels,model):
    if input_points == []:return np.array([])
    predictor, mask_generator = get_model(model)
    predictor.set_image(im)
    input_labels = np.array(input_labels)
    input_points = np.array(input_points)
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False,
    )
    torch.cuda.empty_cache()

    return masks

@st.cache_data
def model_predict_box(im,center_point,center_label,input_box,model):
    predictor, mask_generator = get_model(model)
    predictor.set_image(im)
    masks = np.array([])
    for i in range(len(center_label)):
        if center_point[i] == []:continue
        center_point_1 = np.array([center_point[i]])
        center_label_1 = np.array(center_label[i])
        input_box_1 = np.array(input_box[i])
        mask, score, logits = predictor.predict(
            point_coords=center_point_1,
            point_labels=center_label_1,
            box=input_box_1,
            multimask_output=False,
        )
        try:
            masks = masks + mask
        except:
            masks = mask

    torch.cuda.empty_cache()
    return masks
