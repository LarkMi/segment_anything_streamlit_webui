
## 1. install streamlit_drawable_canvas 
```
pip uninstall streamlit-drawable-canvas
cd streamlit_dc/frontend
npm install
npm run build

pip install -e .

npm run start
```

## 2. run streamlit
```
mkdir checkpoint
cd checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O sam_vit_b_01ec64.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -O sam_vit_l_0b3195.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O sam_vit_h_4b8939.pth

cd ../
streamlit run sam_st.py
```