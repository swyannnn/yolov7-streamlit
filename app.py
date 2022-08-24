import os
import streamlit as st
from detect import detect
from map import map

weight_url = "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt"
@st.cache(show_spinner=False)
def wget_yolov7(weight_url):
    if not os.path.exists('yolov7.pt'):
        os.system(f"wget {weight_url}")
wget_yolov7(weight_url)

def save_uploadedfile(uploadedfile):
    with open(os.path.join("Inference/current.jpg"),"wb") as f:
         f.write(uploadedfile.getbuffer())
    return True
    
def main():
    #remove previous saved image in Inference folder if any
    for file in os.listdir('Inference'):
        if file.endswith('.jpg'):
            os.remove('Inference/'+file) 

    # User interface
    st.title("Scanning electronic items")

    image, camera = st.tabs(["Image", "Camera"])
    
    # if user choose 'Image'
    with image:
        image_file = st.file_uploader("Upload an image",type=["png","jpg","jpeg"])
        if image_file is not None:
            save_uploadedfile(image_file)
            # process and display result image, return bbox_count
            bbox_count = detect(image_file)
            # map features
            map(bbox_count)
    
    # if user choose 'Camera'
    with camera:
        img_file_buffer = st.camera_input("Take a picture")
        if img_file_buffer:
            save_uploadedfile(img_file_buffer)
            # process and display result image, return bbox_count
            bbox_count = detect(img_file_buffer)
            # map features
            map(bbox_count)



if __name__ == "__main__":
    main()
