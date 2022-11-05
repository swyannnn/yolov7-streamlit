from streamlit_option_menu import option_menu
import streamlit as st
from deta import Deta
import streamlit.components.v1 as components
from streamlit_lottie import st_lottie
import json
from PIL import Image
from general import user_status
from general import Deta
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
from streamlit_lottie import st_lottie_spinner
import haversine as hs
from streamlit_lottie import st_lottie
import PIL.Image
import PIL.ExifTags
import json
from PIL import Image
import requests
from django.utils.html import format_html

st.set_page_config(
    page_title="E-waste",
    page_icon="♻️",
)

with st.sidebar:
    choose = option_menu("Welcome", ["Introduction", "Recycle", "Leaderboard"],
                         icons=['emoji-smile', 'recycle', 'clipboard-data'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )

if choose == "Leaderboard":
    user_status()
    # deta = Deta(st.secrets["deta_key"])
    DETA_KEY="c02438ym_H2yB9nr6ho7bCBabFs8D8ecLLqTnpy5C"
    # Initialize with a project key
    deta = Deta(DETA_KEY)
    # This is how to create/connect a database
    db = deta.Base("users_db")
    
    class Deta():
        def fetch_all_users():
            """Returns a dict of all users"""
            res = db.fetch()
            return res.items
    
    users = Deta.fetch_all_users()

    keys = [user["key"] for user in users]
    points = [user["point"] for user in users]
    df = pd.DataFrame({'Username': keys, 'Point': points})
    st.table(df)

if choose == "Introduction":
    user_status()

    def load_lottiefile(filepath: str):
        with open(filepath, "r") as f:
            return json.load(f)

    # FAQ1------------------------------------------------------------------------------------------------------
    def display_lottie_electronics():
        lottie_phone1 = load_lottiefile("./lottiefiles/97008-phone-3d.json")
        lottie_laptop1 = load_lottiefile("./lottiefiles/23662-laptop-animation-pink-navy-blue-white.json")
        lottie_microwaveoven1 = load_lottiefile("./lottiefiles/3139-microwave-oven.json")
        lottie_tv1 = load_lottiefile("./lottiefiles/49117-tv-bad-weather.json")
        lottie_washingmachine1 = load_lottiefile("./lottiefiles/3138-washing-machine.json")
        lottie_ac1 = load_lottiefile("./lottiefiles/75551-air-conditioner.json")
        col1, col2, col3=st.columns((2,2,2))
        with col1:
            st_lottie(lottie_phone1, height=100, key="phone1")
            st_lottie(lottie_tv1, height=100, key="tv1")
        with col2:
            st_lottie(lottie_laptop1, height=100, key="laptop1")
            st_lottie(lottie_washingmachine1, height=100, key="washingmachine1")
        with col3:
            st_lottie(lottie_microwaveoven1, height=100, key="microwaveoven1")
            st_lottie(lottie_ac1, height=100, key="ac1")
    def display_ewaste_barchart():
        HtmlFile = open("ewastebar.html", "r")
        source_code = HtmlFile.read() 
        components.html(source_code, height = 600,width=900)

    st.subheader("What is E-waste?")
    st.write('When a electric electronic appliance is old, broken or non-working, we called it an "E-waste".')
    st.write('These are the examples of electronic items:')
    display_lottie_electronics()
    st.write("Play around with the slider to see the amount of worldwide E-waste genarated over the years!📈")
    display_ewaste_barchart()
    st.write("In Malaysia, we produces more than 365,000 tonnes of e-waste every single year — That's heavier than the weight of the Petronas Twin Towers! Based on research, estimation shows Malaysia generates 24.5 million units of E-waste in 2025. (That's a lot!🤯)")
    # FAQ1------------------------------------------------------------------------------------------------------

    # FAQ2------------------------------------------------------------------------------------------------------
    def display_precious_components_image():
        gold, silver, copper, palladium=st.columns((2,2,2,2))
        with gold:
            image = Image.open('images/precious/gold.png')
            st.image(image, caption='Gold', width=100)
        with silver:
            image = Image.open('images/precious/silver.png')
            st.image(image, caption='Silver', width=100)
        with copper:
            image = Image.open('images/precious/copper.png')
            st.image(image, caption='Copper', width=100)
        with palladium:
            image = Image.open('images/precious/palladium.png')
            st.image(image, caption='Palladium', width=100)
    def display_toxic_components_image():
        arsenic, beryllium, bromine, cadmium, lead, mercury=st.columns((2,2,2,2,2,2))
        with arsenic:
            image = Image.open('images/toxic/arsenic.png')
            st.image(image, caption='Arsenic', width=100)
        with beryllium:
            image = Image.open('images/toxic/beryllium.png')
            st.image(image, caption='Beryllium', width=100)
        with bromine:
            image = Image.open('images/toxic/bromine.png')
            st.image(image, caption='Bromine', width=100)
        with cadmium:
            image = Image.open('images/toxic/cadmium.png')
            st.image(image, caption='Cadmium', width=100)
        with lead:
            image = Image.open('images/toxic/lead.png')
            st.image(image, caption='Lead', width=100)
        with mercury:
            image = Image.open('images/toxic/mercury.png')
            st.image(image, caption='Mercury', width=100)

    st.subheader("What does an E-waste normally contain?")
    st.write("Component in E-waste contains *precious* and *valuable* materials such as:")
    display_precious_components_image()
    st.write("However, E-waste contains *toxic* and *hazardous* materials such as:")
    display_toxic_components_image()
    # FAQ2------------------------------------------------------------------------------------------------------

    # FAQ3------------------------------------------------------------------------------------------------------
    lottie_recycle_icon = load_lottiefile("./lottiefiles/54940-recycle-icon-animation.json")
    lottie_recycle_text = load_lottiefile("./lottiefiles/115879-recycle-text-animation.json")
    st.subheader("How can we reduce the usage of the *toxic* and *hazardous* materials?")
    col1,col2=st.columns((3,7))
    with col1:
        st_lottie(lottie_recycle_text, height=120, key="recycle_text")
        st_lottie(lottie_recycle_icon, height=150, key="recycle_icon")
    with col2:
        st.write("Recycling helps reduce greenhouse gas emissions by reducing energy consumption.")
        st.write("Using recycled materials to make new products reduces the need for virgin materials.")
        st.write("This avoids greenhouse gas emissions that would result from extracting or mining virgin materials.")     
        components.html(
    """
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    div {
    width: 420px;
    padding: 5px;
    border: 3px solid gray;
    margin: 0;
    }
    </style>
    </head>
    <body>
    <h2>Do You Know That</h2>
    <div>Manufacturing products from recycled materials typically requires <b>less energy</b> than making products from virgin materials,<br>which means lesser <b>CARBON FOOTPRINT</b>!!🥳</div>

    </body>
    </html>""")
    # FAQ3------------------------------------------------------------------------------------------------------

    # FAQ4------------------------------------------------------------------------------------------------------
    lottie_say_no_CO2 = load_lottiefile("./lottiefiles/33545-carbon-dioxide-emission.json")
    st.subheader("Carbon Footprint")
    col1,col2=st.columns((2,8))
    with col1:
        st_lottie(lottie_say_no_CO2, height=100, key="say_no_CO2")
        image = Image.open('images/others/Earth-Footprints.png')
        st.image(image, width=90)
    with col2:
        st.write("Carbon footprint is the total amount of greenhouse gases that are caused by the choices and actions of an individual, company or a nation.")
        st.write("Carbon footprint is measured in terms of carbon dioxide emissions (CO2).")
        st.write("Carbon Footprint per person in Malaysia are equivalent to 8.68 tons per person. Globally, the average carbon footprint is closer to 4 tons.")
        st.write("To have the best chance of avoiding a 2°C rise in global temperatures, the average global carbon footprint per year needs to drop to under 2 tons by 2050.")
    # FAQ4------------------------------------------------------------------------------------------------------

    # FAQ5------------------------------------------------------------------------------------------------------
    lottie_questionman = load_lottiefile("./lottiefiles/32045-question.json")
    col1,col2=st.columns((8,2))
    with col1:
        st.subheader("We should recycle the E-waste instead of throwing it into rubbish bin. WHY??")
        st.write("E-waste is becoming a global issue🌏")
        st.write("The more electrical and electronic equipment are being produced, the more E-waste need to be disposed or managed properly.")
        st.write("If e-waste is discarded without implementing environmentally sound manner such as into the river, landfill, burning or sent to informal sector, e-waste may endanger our life, affecting human health and causing deterioration of environmental quality.")
        st.write("Therefore, we should properly managed e-waste in environmentally sound manner!")
    with col2:
        st_lottie(lottie_questionman, height=200, key="questionman")
    
if choose == "Recycle":
    user_status()
    def centredata():
        centres = pd.read_csv('centredata2.csv')

        map = folium.Map(location=[centres.Latitude.mean(), centres.Longitude.mean()], zoom_start=7, tiles='OpenStreetMap')

        # loop all centres in csv file and plot locations on map
        for _, centre in centres.iterrows():
            folium.Marker(
                location=[centre['Latitude'], centre['Longitude']],
                popup=centre['CompanyName'],
                tooltip=centre['CompanyName'],
                icon=folium.Icon(color="green",icon="fa-recycle", prefix='fa')
            ).add_to(map)
        
        st_folium(map)
    # create a button to access user's location
    def permissionbutton():
        loc_button = Button(label="Direct me to the top 5 nearest E-waste collection centre")
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
            override_height=50,
            debounce_time=0)
        return result
    # getting user's location when he/she allows 
    def getuserlocation(result):
        latitude = result.get("GET_LOCATION")['lat']
        longitude = result.get("GET_LOCATION")['lon']
        return latitude,longitude
    # plot user's location on map
    def plotuserlocation(latitude,longitude):
        map = folium.Map(width=10,height=10,location=(latitude,longitude),zoom_start = 12)

        # user's location as centre on map
        folium.Marker([latitude,longitude], popup = f"Your location",
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
    # final algorithm using the functions above for map
    def func_for_map_feature():
        result = permissionbutton()
        if result:
            latitude,longitude = getuserlocation(result)
            map = plotuserlocation(latitude,longitude)
            st.subheader('Top 5 nearest recycle centre from your current location (Informations attached below)')
            link = 'Click [HERE](https://ewaste.doe.gov.my/index.php/about/list-of-collectors/) to know more details about all the government proved E-waste recycling centre in Malaysia'
            st.markdown(link,unsafe_allow_html=True)
            centre_loc = nearestcentre(map,latitude,longitude)
            st_folium(map)
            listnearestcentre(centre_loc)
        else:
            st.subheader("Government proved recycling centre in Malaysia")
            link = 'Click [HERE](https://ewaste.doe.gov.my/index.php/about/list-of-collectors/) to know more details about all the government proved E-waste recycling centre in Malaysia'
            st.markdown(link,unsafe_allow_html=True)
            centredata()
        return True

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
    # complete detect process
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
            if found is not None:
                # display image in st
                im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                st.image(im0)
                # save detected result image into "Inference" folder
                im = Image.fromarray(im0)
                im.save("Inference/result.jpg")
                download_result_image()
        return found

    # Load the environment variables
    deta2 = Deta(st.secrets["deta_key2"])
    # This is how to create/connect a database
    db2 = deta2.Base("users_submit")

    # Load the environment variables
    deta3 = Deta(st.secrets["deta_key3"])
    # This is how to create/connect a database
    db3 = deta3.Drive("users_submit")

    def submit_item_form(found):
        submit_item_form = st.form('Submit Item')
        submit_item_form.subheader('I want to recycle this item')
        # username = submit_item_form.text_input('Username').lower()
        yesorno =  submit_item_form.radio(
            "Did I detect your electronic item correctly?",
            (f'Yes💯, it is {found}!', 'No☹️'))
        if_no = submit_item_form.selectbox('If no, may I know what is it?',['-',
                                'Keyboard',
                                'Cell Phone',
                                'Laptop',
                                'Hair Dryer',
                                'Toaster',
                                'Oven',
                                'Microwave',
                                'TV',
                                'Mouse',
                                'Remote',
                                'Refridgerator',
                                'Camera',
                                'Printer',
                                'Washing Machine'
                                'Others'])
        if submit_item_form.form_submit_button('Submit'):
            db2.put({"key": st.session_state['key'], "accuracy": yesorno, "if_no": if_no})
            if found is not None:
                db3.put(f"{st.session_state['key']}--correct.jpg", path="Inference/result.jpg")
                current_point = Deta.get_user(f"{st.session_state['key']}")['point']
                Deta.update_user(f"{st.session_state['key']}", updates={"point":current_point+2})

            else:
                db3.put(f"{st.session_state['key']}--wrong--{if_no}.jpg", path="Inference/current.jpg")

    # Load the environment variables
    deta2 = Deta(st.secrets["deta_key2"])
    # This is how to create/connect a database
    db2 = deta2.Base("users_submit")

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

    def download_result_image():
        with open("Inference/result.jpg", "rb") as file:
            btn = st.download_button(
                    label="Download image",
                    data=file,
                    file_name="result.jpg",
                    mime="image/png")
            return btn

    # detect whether there is electronic items in the image using "detect" function
    # if nothing found, st.text('nothing detected')
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
        return found

    def feedback_form():
        st.header(":mailbox: Get In Touch With Me!")

        contact_form = """
        <form action="https://formsubmit.co/seow.w22@kinghenryviii.edu.my" method="POST">
            <input type="hidden" name="_captcha" value="false">
            <input type="text" name="name" placeholder="Your name" required>
            <input type="email" name="email" placeholder="Your email" required>
            <textarea name="message" placeholder="Your message here"></textarea>
            <button type="submit">Send</button>
        </form>
        """

        st.markdown(contact_form, unsafe_allow_html=True)

        # Use Local CSS File
        def local_css(file_name):
            with open(file_name) as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


        local_css("style/style.css")

    @st.experimental_memo
    def load_lottiefile(filepath: str):
        with open(filepath, "r") as f:
            return json.load(f)
    @st.experimental_memo
    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()

    def main():
        #remove previous saved image in Inference folder if any
        for file in os.listdir('Inference'):
            if file.endswith('.jpg'):
                os.remove('Inference/'+file) 

        # User interface
        if st.session_state['key'] is not None:
            st.write(f"You are now logged in as {st.session_state['key']}")
        lottie_col, title_col= st.columns((2,8))
        with lottie_col:
            lottie_recycle_icon2 = load_lottiefile("./lottiefiles/120673-goods-recycling.json")
            st_lottie(lottie_recycle_icon2, height=100, key="recycle_icon2")
        with title_col:
            st.title("Ready to Recycle")
        upload_col, map_col = st.columns(2)
        with upload_col:
            st.write('Guidance to gain point(s) from recycling your E-waste:')
            st.write('3) Make sure you have signed in your account.')
            st.write('2) Make sure you are taking picture around any E-waste collection centre listed in the map. (Because we will need to verify your location)')
            st.write('3) Take a CLEAR  picture of your E-waste')
            st.write('4) Fill in the submit form.')
            st.write('5) You are done! We will update your game point(s) in 48 hours.')
            camera_image_file = st.camera_input("📸Take a picture")
            image_file = st.file_uploader("📤Upload an image",type=["png","jpg","jpeg"])
            if camera_image_file:
                found = detectionprocess(camera_image_file)
                submit_item_form(found)
            if image_file is not None:
                found = detectionprocess(image_file)
                submit_item_form(found)
            

        with map_col:
            func_for_map_feature()
        feedback_form()

    if __name__ == "__main__":
        main()

