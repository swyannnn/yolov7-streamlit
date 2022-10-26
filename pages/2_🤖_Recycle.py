import streamlit as st
import os
import argparse
import cv2
import torch
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, \
    scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, TracedModel
from streamlit_folium import st_folium
# map import
import folium
import pandas as pd
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events
import haversine as hs
from streamlit_lottie import st_lottie
import PIL.Image
import PIL.ExifTags

st.set_page_config(
    page_title="E-waste",
    page_icon="♻️",
)

# before getting location access permission from user, plot all centre locations on map
def centredata():
    centres = pd.read_csv('centredata2.csv')

    map = folium.Map(location=[centres.Latitude.mean(), centres.Longitude.mean()], zoom_start=7, tiles='OpenStreetMap')

    # loop all centres in csv file and plot locations on map
    for _, centre in centres.iterrows():
        folium.Marker(
            location=[centre['Latitude'], centre['Longitude']],
            popup=centre['CompanyName'],
            tooltip=centre['CompanyName'],
            icon=folium.Icon(color='darkgreen', icon_color='white',prefix='fa', icon='circle')
        ).add_to(map)
    
    st_folium(map)
# create a button to access user's location
def permissionbutton():
    loc_button = Button(label="Direct me to nearest centre")
    loc_button.js_on_event("button_click", CustomJS(code="""
        navigator.geolocation.getCurrentPosition(
            (loc) => {
                document.dispatchEvent(new CustomEvent("GET_LOCATION", {detail: {lat: loc.coords.latitude, lon: loc.coords.longitude}}))
            }
        )
        """))

    result = streamlit_bokeh_events(
        loc_button,
        events="GET_LOCATION",
        key="get_location",
        refresh_on_update=False,
        override_height=75,
        debounce_time=0)
    return result
# getting user's location when he/she allows 
def getuserlocation(result):
    latitude = result.get("GET_LOCATION")['lat']
    longitude = result.get("GET_LOCATION")['lon']
    return latitude,longitude
# plot user's location on map
def plotuserlocation(latitude,longitude):
    map = folium.Map(width=10,height=10,location=(latitude,longitude),zoom_start = 15)

    # user's location as centre on map
    folium.Marker([latitude,longitude], popup = f"Your location:{latitude},{longitude}",
    tooltip=f"Your location:{latitude},{longitude}", icon=folium.Icon(color="red",icon="fa-home", prefix='fa')).add_to(map)

    return map
# find and plot nearest centre from user
def nearestcentre(map,latitude,longitude):
    # read csv file
    centre_loc=pd.read_csv('centredata2.csv')

    # zip data for each column
    centre_loc['coor'] = list(zip(centre_loc.Latitude, centre_loc.Longitude))

    # function to obtain distance between user's and centre's locations
    def distance_from(loc1,loc2): 
        distance=hs.haversine(loc1,loc2)
        return round(distance,1)

    # make a list to record the distances
    distance = list()
    for _,row in centre_loc.iterrows():
        distance.append(distance_from(row.coor,(latitude,longitude)))

    # assigning data in list to each columns
    centre_loc['distance']=distance

    centre_loc = centre_loc.sort_values(by=['distance'])

    # plotting the 5 nearest centre from user's location on map
    x = 1
    for index, row in centre_loc.iterrows(): 
        if x <= 5:
            folium.Marker(
                location= [row['Latitude'],row['Longitude']],
                radius=5,
                popup= f"{row['CompanyName']}({row['distance']}km)",
                tooltip=f"{row['CompanyName']}({row['distance']}km)",
                color='red',
                fill=True,
                fill_color='red',
                icon=folium.Icon(color="green",icon="fa-recycle", prefix='fa')
                ).add_to(map)
            x+=1
    return centre_loc
# listing the 5 nearest centre from user's location on map
def listnearestcentre(centre_loc):
    x = 1
    for index, row in centre_loc.iterrows(): 
        if x <= 5:
            st.write(f"""
            {x}) {row["CompanyName"]} -- {row['distance']} km\n
            📍 {row["Address"]}\n
            :telephone_receiver: {row['TelNum']}\n
            """)
            x+=1
    link = 'Click [HERE](https://ewaste.doe.gov.my/index.php/about/list-of-collectors/) to know all the government proved recycling centre in Malaysia'
    st.markdown(link,unsafe_allow_html=True)
# final algorithm using the functions above for map
def func_for_map_feature():
    result = permissionbutton()
    if result:
        latitude,longitude = getuserlocation(result)
        map = plotuserlocation(latitude,longitude)
        st.subheader('Top 5 nearest recycle centre from your current location')
        centre_loc = nearestcentre(map,latitude,longitude)
        st_folium(map)
        listnearestcentre(centre_loc)
    else:
        centredata()

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

def detect(img, weight_file):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=f"{weight_file}.pt", help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='Inference/', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.6, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--trace', action='store_true', help='trace model')
    opt = parser.parse_args()
    source, weights, imgsz, trace = opt.source, opt.weights, opt.img_size, opt.trace

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
    

    found = None
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

        if weights == "yolov7.pt":
            target = ['tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                            'microwave', 'oven', 'toaster', 'refrigerator', 'hair drier']
            for i, det in enumerate(pred):
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    max_conf = 0
                    max_label = {}
                    for *xyxy, conf, cls in reversed(det):
                        label = names[int(cls)]
                        if label in target and conf > max_conf:
                            max_conf = conf
                            max_label = {label:xyxy}
                    if len(max_label) > 0:
                        st.text(max_conf)
                        for key, value in max_label.items():
                            found = key
                            plot_one_box(value, im0, label=f"{key}", color=[0,255,0], line_thickness=2)
                            st.text(key)
                             
            

        else:
            for i, det in enumerate(pred):
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    for *xyxy, conf, cls in reversed(det):
                        found = weight_file
                        plot_one_box(xyxy, im0, label=weight_file, color=[0,255,0], line_thickness=2)

        # display image in st
        if found is not None:
            im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
            st.image(im0)
        return found
        
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

def detectionprocess(image_file):
    save_uploadedfile(image_file)
    # process and display result image, return bbox_count
    found = detect(image_file, "yolov7")
    pt_list = ['washingmachine','camera','printer']
    pt_index = 0
    while found is None and pt_index < len(pt_list):
        found = detect(image_file, pt_list[pt_index])
        pt_index += 1
    if found is None and pt_index == len(pt_list):
        st.text('nothing detected')

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

def main():
    #remove previous saved image in Inference folder if any
    for file in os.listdir('Inference'):
        if file.endswith('.jpg'):
            os.remove('Inference/'+file) 

    # User interface
    st.title("Ready to Recycle")

    upload_col, map_col = st.columns(2)
    with upload_col:
        ottie_upload = load_lottiefile("./lottiefiles/565-camera.json")
        st_lottie(ottie_upload, height=10, key="upload_icon")
        camera_image_file = st.camera_input("Take a picture")
        image_file = st.file_uploader("Upload an image",type=["png","jpg","jpeg"])
        if image_file is not None:
            detectionprocess(image_file)
        if camera_image_file:
            detectionprocess(camera_image_file)
    with map_col:
        func_for_map_feature()
        # if image_file is not None or camera_image_file:
        #     img = PIL.Image.open("Inference/current.jpg")
        #     exif = {
        #         PIL.ExifTags.TAGS[key]: value
        #         for key, value in img._getexif().items()
        #         if key in PIL.ExifTags.TAGS
        #     }
        #     north = exif["GPSInfo"][2]
        #     east = exif["GPSInfo"][4]
        #     latitude = float((north[0]) + (north[1]/60) + (north[2]/3600))
        #     longitude = float((east[0]) + (east[1]/60) + (east[2]/3600))
        #     map = plotuserlocation(latitude,longitude)
        #     centre_loc = nearestcentre(map,latitude,longitude)
        #     st.subheader('Top 5 nearest recycle centre from your current location')
        #     nearestcentre(centre_loc)
        #     st_folium(map)
        #     listnearestcentre(centre_loc)


if __name__ == "__main__":
    main()


# #test image location
# img = PIL.Image.open("Inference/current.jpg")
# exif = {
#     PIL.ExifTags.TAGS[key]: value
#     for key, value in img._getexif().items()
#     if key in PIL.ExifTags.TAGS
# }
# north = exif["GPSInfo"][2]
# east = exif["GPSInfo"][4]
# latitude = float((north[0]) + (north[1]/60) + (north[2]/3600))
# longitude = float((east[0]) + (east[1]/60) + (east[2]/3600))
# map = plotuserlocation(latitude,longitude)
# centre_loc = nearestcentre(map,latitude,longitude)
# st.subheader('Top 5 nearest recycle centre from your current location')
# nearestcentre(centre_loc)
# st_folium(map)
# listnearestcentre(centre_loc)
