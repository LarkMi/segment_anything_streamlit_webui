
## 1. install streamlit_drawable_canvas 
```
pip uninstall streamlit-drawable-canvas
cd streamlit_dc/streamlit_drawable_canvas/frontend
npm install
npm run build
cd streamlit_dc/
pip install -e .
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

# Docker
```
docker build -t sam_st .
docker run --gpus "device=1" -itd -p 84:8501 --name sam_st sam_st:latest bash -c 'streamlit run sam_st.py'
```