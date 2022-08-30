import streamlit as st
import os, argparse, cv2, torch
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, \
    scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, TracedModel
from streamlit_folium import st_folium

st.set_page_config(
    page_title="E-waste",
    page_icon="♻️",
)

# Initialize
@st.cache(show_spinner=False)
def initialize(device):
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    return device,half

# Load model
@st.cache(show_spinner=False)
def loadmodel(weights,device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    return model,stride

# get names
@st.cache(show_spinner=False)
def getnames(model):
    names = model.module.names if hasattr(model, 'module') else model.names
    return names

def detect(img):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default="yolov7.pt", help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='Inference/', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.6, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--trace', action='store_true', help='trace model')
    opt = parser.parse_args()
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.trace

    # Initialize
    device,half = initialize(opt.device)

    # Load model
    model,stride = loadmodel(weights,device)
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)
    
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names
    names = getnames(model)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        for i, det in enumerate(pred):
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Add bbox to image
                bbox_count = 0
                for *xyxy, conf, cls in reversed(det):
                    label = names[int(cls)]
                    target = ['tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                            'microwave', 'oven', 'toaster', 'refrigerator', 'hair drier']
                    if label in target:
                        plot_one_box(xyxy, im0, label=label, color=[0,255,0], line_thickness=2)
                        bbox_count += 1    

        # display image in st
        im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
        st.image(im0)

    return bbox_count
        
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
            # map(bbox_count)
            if bbox_count>0:
                st.write('congratz')
                map = st.button('direct me to nearest centre')
                if map:
                    link = 'http://localhost:8501/Map'
                    st.markdown(link, unsafe_allow_html=True)
    
    # if user choose 'Camera'
    with camera:
        img_file_buffer = st.camera_input("Take a picture")
        if img_file_buffer:
            save_uploadedfile(img_file_buffer)
            # process and display result image, return bbox_count
            bbox_count = detect(img_file_buffer)
            # map features
            if bbox_count>0:
                st.write('congratz')
                map = st.button('direct me to nearest centre')
                if map:
                    link = 'http://localhost:8501/Map'
                    st.markdown(link, unsafe_allow_html=True)
            # map(bbox_count)



if __name__ == "__main__":
    main()
