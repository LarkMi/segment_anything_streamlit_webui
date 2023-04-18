import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import streamlit as st
import torch
from PIL import Image
import numpy as np
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import random
from io import BytesIO
from util import model_predict_click, model_predict_box, model_predict_everything, show_click, show_everything, get_color


def main():
    print('init')
    torch.cuda.empty_cache()
    with st.sidebar:
        im = st.file_uploader(label='Upload image',type=['png','jpg','tif'])
        option = st.selectbox(
            'Segmentation mode',
            ('Click', 'Box', 'Everything'))
        model = st.selectbox(
            'Model',
            ('vit_b', 'vit_l', 'vit_h'))
        show_mask = st.checkbox('Show mask',value = True)
        radius_width = st.slider('Radius/Width for Click/Box',0,20,5,1)
        
    if im:
        im = Image.open(im).convert("RGB")
        if 'im' not in st.session_state:
            st.session_state['im'] = im
        width, height   = im.size[:2]
        im              = np.array(im)
        container_width = 700
        scale           = container_width/width
        if option == 'Click':
            for each in ['color_change_point_box','input_masks_color_box']:
                if each in st.session_state:st.session_state.pop(each)
            canvas_result = st_canvas(
                    fill_color="rgba(255, 255, 0, 0.8)",
                    background_image = st.session_state['im'],
                    drawing_mode='point',
                    width = container_width,
                    height = height * scale,
                    point_display_radius = radius_width,
                    stroke_width=2,
                    update_streamlit=True,
                    key="click",)
            if not show_mask:
                im = Image.fromarray(im).convert("RGB")
                rerun = False
                if im != st.session_state['im']:
                    rerun = True
                st.session_state['im'] = im
                if rerun:
                    st.experimental_rerun()
            elif canvas_result.json_data is not None:
                color_change_point = st.button('Save color')
                df = pd.json_normalize(canvas_result.json_data["objects"])
                if len(df) == 0:
                    st.session_state.clear()
                    if 'canvas_result' not in st.session_state:
                        st.session_state['canvas_result'] = len(df)
                        st.experimental_rerun()
                    elif len(df) != st.session_state['canvas_result']:
                        st.session_state['canvas_result'] = len(df)
                        st.experimental_rerun()
                    return
                
                df["center_x"] = df["left"]
                df["center_y"] = df["top"]
                input_points = []
                input_labels = []
                
                for _, row in df.iterrows():
                    x, y = row["center_x"] + 5, row["center_y"]
                    x = int(x/scale)
                    y = int(y/scale)
                    input_points.append([x, y])
                    if row['fill'] == "rgba(0, 255, 0, 0.8)":
                        input_labels.append(1)
                    else:
                        input_labels.append(0)
                
                if 'color_change_point' in st.session_state:
                    p = st.session_state['color_change_point']
                    if len(df) < p:
                        p = len(df) - 1
                        st.session_state['color_change_point'] = p
                    masks = model_predict_click(im,input_points[p:],input_labels[p:],model)
                else:
                    masks = model_predict_click(im,input_points,input_labels,model)
                
                if color_change_point:
                    st.session_state['color_change_point'] = len(df)
                    st.session_state['input_masks_color'].append([np.array([]),np.array([])])
                else:
                    color = np.concatenate([random.choice(get_color()), np.array([0.6])], axis=0)
                    if 'input_masks_color' not in st.session_state:
                        st.session_state['input_masks_color'] = [[masks,color]]
                    
                    elif not np.array_equal(st.session_state['input_masks_color'][-1][0],masks):
                        st.session_state['input_masks_color'][-1] = [masks,color]
                    im_masked = show_click(st.session_state['input_masks_color'])
                    im_masked = Image.fromarray(im_masked).convert('RGBA')
                    im = Image.alpha_composite(Image.fromarray(im).convert('RGBA'),im_masked).convert("RGB")
                    torch.cuda.empty_cache()
                    rerun = False
                    if im != st.session_state['im']:
                        rerun = True
                    st.session_state['im'] = im
                    if rerun:
                        st.experimental_rerun()
                im_bytes = BytesIO()
                st.session_state['im'].save(im_bytes,format='PNG')
                st.download_button('Download image',data=im_bytes.getvalue(),file_name='seg.png')
            
        if option == 'Box':
            
            for each in ['color_change_point','input_masks_color']:
                if each in st.session_state:st.session_state.pop(each)
            
            canvas_result_1 = st_canvas(
                    fill_color="rgba(255, 255, 0, 0)",
                    background_image = st.session_state['im'],
                    drawing_mode='rect',
                    stroke_color = "rgba(0, 255, 0, 0.6)",
                    stroke_width = radius_width,
                    width = container_width,
                    height = height * scale,
                    point_display_radius = 12,
                    update_streamlit=True,
                    key="box",
                    )
            if not show_mask:
                im = Image.fromarray(im).convert("RGB")
                rerun = False
                if im != st.session_state['im']:
                    rerun = True
                st.session_state['im'] = im
                if rerun:
                    st.experimental_rerun()
            elif canvas_result_1.json_data is not None:
                color_change_point = st.button('Save color')
                df = pd.json_normalize(canvas_result_1.json_data["objects"])
                if len(df) == 0:
                    st.session_state.clear()
                    if 'canvas_result' not in st.session_state:
                        st.session_state['canvas_result'] = len(df)
                        st.experimental_rerun()
                    elif len(df) != st.session_state['canvas_result']:
                        st.session_state['canvas_result'] = len(df)
                        st.experimental_rerun()
                    return
                center_point,center_label,input_box = [],[],[]
                for _, row in df.iterrows():
                    x, y, w,h = row["left"], row["top"], row["width"], row["height"]
                    x = int(x/scale)
                    y = int(y/scale)
                    w = int(w/scale)
                    h = int(h/scale)
                    center_point.append([x+w/2,y+h/2])
                    center_label.append([1])
                    input_box.append([x,y,x+w,y+h])
                #masks, scores = model_predict_box(im,center_point,center_label,input_box,model)
                #im_masked = show_click(masks,scores)
                
                if 'color_change_point_box' in st.session_state:
                    p = st.session_state['color_change_point_box']
                    if len(df) < p:
                        p = len(df) - 1
                        st.session_state['color_change_point_box'] = p
                    masks = model_predict_box(im,center_point[p:],center_label[p:],input_box[p:],model)
                else:
                    masks = model_predict_box(im,center_point,center_label,input_box,model)
                masks = np.array(masks)
                if color_change_point:
                    st.session_state['color_change_point_box'] = len(df)
                    st.session_state['input_masks_color_box'].append([np.array([]),np.array([])])
                else:
                    color = np.concatenate([random.choice(get_color()), np.array([0.6])], axis=0)
                    if 'input_masks_color_box' not in st.session_state:
                        st.session_state['input_masks_color_box'] = [[masks,color]]
                    
                    elif not np.array_equal(st.session_state['input_masks_color_box'][-1][0],masks):
                        st.session_state['input_masks_color_box'][-1] = [masks,color]
                    im_masked = show_click(st.session_state['input_masks_color_box'])
                    im_masked = Image.fromarray(im_masked).convert('RGBA')
                    im = Image.alpha_composite(Image.fromarray(im).convert('RGBA'),im_masked).convert("RGB")
                    torch.cuda.empty_cache()
                    rerun = False
                    if im != st.session_state['im']:
                        rerun = True
                    st.session_state['im'] = im
                    if rerun:
                        st.experimental_rerun()
                    im_bytes = BytesIO()
                    st.session_state['im'].save(im_bytes,format='PNG')
                    st.download_button('Download image',data=im_bytes.getvalue(),file_name='seg.png')
                
            
        if option == 'Everything':
            everything = st.image(Image.fromarray(im))
            if  show_mask:

                masks = model_predict_everything(im,model)
                im_masked = show_everything(masks)
                im_masked = Image.fromarray(im_masked).convert('RGBA')
                im = Image.alpha_composite(Image.fromarray(im).convert('RGBA'),im_masked).convert("RGB")
                everything.image(im)
                torch.cuda.empty_cache()
                im_bytes = BytesIO()
                im.save(im_bytes,format='PNG')
                st.download_button('Download image',data=im_bytes.getvalue(),file_name='seg.png')

    else:
        st.session_state.clear()

if __name__ == '__main__':
    main()