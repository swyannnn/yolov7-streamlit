import os
import streamlit as st
import argparse
from pathlib import Path
# os.system("wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt")
import cv2, torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

def main():
    #remove previous image
    for file in os.listdir('Inference'):
            os.remove('Inference/'+file) 

    # User interface
    st.title("Scanning electronic items")

    image, camera = st.tabs(["Image", "Camera"])
    
    # if user choose 'Image'
    with image:
        image_file = st.file_uploader("Upload an image",type=["png","jpg","jpeg"])
        if image_file is not None:
            save_uploadedfile(image_file)
            detect(image_file)
    
    # if user choose 'Camera'
    with camera:
        img_file_buffer = st.camera_input("Take a picture")
        if img_file_buffer:
            save_uploadedfile(img_file_buffer)
            detect(img_file_buffer)

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
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--trace', action='store_true', help='trace model')
    opt = parser.parse_args()
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.trace

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
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

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

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
                for *xyxy, conf, cls in reversed(det):
                    label = names[int(cls)]
                    target = ['tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                            'microwave', 'oven', 'toaster', 'refrigerator', 'hair drier']
                    if label in target:
                        plot_one_box(xyxy, im0, label=label, color=[0,255,0], line_thickness=3)


        # display image in st
        im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
        st.image(im0)

                    
def save_uploadedfile(uploadedfile):
    #  with open(os.path.join("Inference",uploadedfile.name),"wb") as f:
    with open(os.path.join("Inference/current.jpg"),"wb") as f:
         f.write(uploadedfile.getbuffer())
    return True


if __name__ == "__main__":
    main()
