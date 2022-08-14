import streamlit as st
import numpy as np
from PIL import Image
import cv2, os, urllib, time
import certifi, ssl

def main():
    # User interface
    st.title("Scanning electronic items")

    option = st.sidebar.selectbox(
        'Option',
        ('Image', 'Camera'))
    
    # Download external dependencies.
    # for filename in EXTERNAL_DEPENDENCIES.keys():
    #     download_file(filename)
    
    # if user choose 'Image'
    if option == 'Image':
        image_file = st.file_uploader("Upload an image",type=["png","jpg","jpeg"])

        # if image_file is not None:
        #     image = Image.open(image_file)
        #     image_BGR = np.array(image)
        #     general(image_BGR)


# # This file downloader demonstrates Streamlit animation.
# def download_file(file_path):
#     # Don't download the file twice. (If possible, verify the download using the file length.)
#     if os.path.exists(file_path):
#         if "size" not in EXTERNAL_DEPENDENCIES[file_path]:
#             return
#         elif os.path.getsize(file_path) == EXTERNAL_DEPENDENCIES[file_path]["size"]:
#             return

#     # These are handles to two visual elements to animate.
#     weights_warning, progress_bar = None, None
#     try:
#         weights_warning = st.warning("Downloading...")
#         progress_bar = st.progress(0)
#         with open(file_path, "wb") as output_file:
#             # with urllib.request.urlopen(EXTERNAL_DEPENDENCIES[file_path]["url"]) as response:
#             with urllib.request.urlopen(EXTERNAL_DEPENDENCIES[file_path]["url"], context=ssl.create_default_context(cafile=certifi.where())) as response:
#                 length = int(response.info()["Content-Length"]) 
#                 counter = 0
#                 # MEGABYTES = 2.0 ** 20.0
#                 while True:
#                     data = response.read(8192)
#                     if not data:
#                         break
#                     counter += len(data)
#                     output_file.write(data)

#                     # We perform animation by overwriting the elements.
#                     weights_warning.warning(f"Downloading... {int((counter/length)*100)}%" )
#                     progress_bar.progress(min(counter / length, 1.0))

#     # Finally, we remove these visual elements by calling .empty().
#     finally:
#         if weights_warning is not None:
#             weights_warning.empty()
#         if progress_bar is not None:
#             progress_bar.empty()


# # yolov3 working included here
# def yolo7(image_RGB):
#     # making list of coco.names (80 elements)
#     with open("coco.yaml") as f:
#         labels = [line.strip() for line in f]

#     # Load the network. Because this is cached it will only happen once.
#     @st.cache(allow_output_mutation=True)
#     def load_network(config_path, weights_path):
#         network = cv2.dnn.readNetFromDarknet(config_path, weights_path)
#         layers_names_all = network.getLayerNames()
#         layers_names_output = [layers_names_all[i - 1] for i in network.getUnconnectedOutLayers()]
#         return network, layers_names_output
#     network, layers_names_output = load_network("yolov4.cfg", "yolov4.weights")

#     # setting requirements to filter detected objects
#     target_items = ['tvmonitor',
#                     'laptop',
#                     'mouse',
#                     'remote',
#                     'keyboard',
#                     'cell phone',
#                     'microwave',
#                     'oven',
#                     'toaster',
#                     'refrigerator',
#                     'hair drier']
#     probability_minimum = 0.5
#     threshold = 0.3

#     # assigning variables for future use
#     bounding_boxes, confidences, class_numbers = [],[],[]
#     h, w = image_RGB.shape[:2]

#     # creating blob & forward pass
#     blob = cv2.dnn.blobFromImage(image_RGB, 1 / 255.0, (416, 416), swapRB=True, crop=False)
#     network.setInput(blob)  
#     output_from_network = network.forward(layers_names_output)

#     # extracting needed information for NMS
#     for result in output_from_network:
#         for detected_objects in result:
#             scores = detected_objects[5:]
#             class_current = np.argmax(scores)
#             confidence_current = scores[class_current]

#             if confidence_current > probability_minimum:
#                 box_current = detected_objects[0:4] * np.array([w, h, w, h])

#                 x_center, y_center, box_width, box_height = box_current
#                 x_min = int(x_center - (box_width / 2))
#                 y_min = int(y_center - (box_height / 2))

#                 bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
#                 confidences.append(float(confidence_current))
#                 class_numbers.append(class_current)
    
#     # Non-maximum Suppression
#     results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
#                                probability_minimum, threshold)

#     # extracting needed information for final output
#     counter = 1
#     description = str()
#     if len(results) > 0:
#         description = str()
#         for i in results.flatten():
#             if (labels[int(class_numbers[i])]) in target_items:
#                 description = description + f'Object {counter}: {labels[int(class_numbers[i])]}' + "\n"

#                 counter += 1

#                 x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
#                 box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

#                 cv2.rectangle(image_RGB, (x_min, y_min),
#                             (x_min + box_width, y_min + box_height),
#                             (0,255,0), 2)

#                 text_box_current = str(labels[int(class_numbers[i])])

#                 cv2.putText(image_RGB, text_box_current, (x_min, y_min - 5),
#                             cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,255,0), 2)


#     return image_RGB, description

# # Function used by both image and camera option
# def general(image_BGR):
#     # A spinner while result image is still in progress
#     with st.spinner('Wait for it...'):
#         time.sleep(5)

#     # connect to yolo3 function and output the final result on streamlit app
#     image_BGR, description = yolo7(image_BGR)

#     if len(description) == 0:
#         st.image(image_BGR, caption='No electronic item in the image')
#     else:
#         st.image(image_BGR,caption='There is electronic item(s) in the image')
#         st.text(description)

# # External files to download.
# EXTERNAL_DEPENDENCIES = {
#     "yolov7.weights": {
#         "url": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt",
#         "size": 141000
#     },
#     "yolov7.yaml": {
#         "url": "https://raw.githubusercontent.com/WongKinYiu/yolov7/main/cfg/training/yolov7.yaml",
#         "size": 3.91
#     },
#     "coco.yaml": {
#         "url": "https://raw.githubusercontent.com/WongKinYiu/yolov7/main/data/coco.yaml",
#         "size": 1.39
#     }
# }

if __name__ == "__main__":
    main()
