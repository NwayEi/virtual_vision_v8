import multiprocessing as mp
from multiprocessing import Process
from ultralytics import YOLO
from gtts import gTTS
import ReferenceImageVal as ri
import signal
import streamlit as st
import numpy as np
import os
import logging
import time
from io import BytesIO
from PIL import Image
import ReferenceImageVal as rf

#DISTANCE CONTASTANT
KNOWN_DISTANCE = 1.5 # meter

#INDOOR
cell_phone_WIDTH = 0.08 #meter
person_WIDTH = 0.40 #meter
backpack_WIDTH = 0.35
handbag_WIDTH = 0.26
chair_WIDTH = 0.5
dining_table_WIDTH = 0.96
laptop_WIDTH = 0.35
bench_WIDTH = 0.45
couch_WIDTH = 2.21
potted_plant_WIDTH = 0.13
suitcase_WIDTH = 0.45
tv_WIDTH = 0.96

#OUTDOOR
car_WIDTH = 1.77
bicycle_WIDTH = 0.75
motorcycle_WIDTH = 0.86
stopsign_WIDTH = 0.6
traffic_light_WIDth= 0.2
parking_meter_WIDTH = 2.7


processes=[]
selected_detected_class = [0,13,26,56,24,57,63,58,62,60,28,1,2,3,9,11,12]
base_model = YOLO('yolov8n.pt')  #yolov8n.pt load a pretrained model (recommended for training)
custom_model = YOLO('baseline_door_working_audio.pt') #custom trained model to classify male and female

def IndoorDetectReferenceImages():

    result_person = base_model.predict(source='ReferenceImages/person.png')
    for j,box in enumerate(result_person[0].boxes):
        if box.cls.item() == 0.0 :
            ri.person_width_in_rf = box.xywh[0][2]
            print(f'-----------Person RF Class Name ID : {box.cls.item()} - WIDTH : {ri.person_width_in_rf}')

    result_chair= base_model.predict(source='ReferenceImages/chair.jpeg')
    for j,box in enumerate(result_chair[0].boxes):
        if box.cls.item() == 56.0 :
            ri.chair_width_in_rf = box.xywh[0][2]
            print(f'-----------Chair RF Class Name ID : {box.cls.item()} - WIDTH : {ri.chair_width_in_rf}')

    result_handbag = base_model.predict(source='ReferenceImages/handbag.jpeg')
    for j,box in enumerate(result_handbag[0].boxes):
        if box.cls.item() == 26.0 :
            ri.handbag_width_in_rf = box.xywh[0][2]
            print(f'-----------HandBag RF Class Name ID : {box.cls.item()} - WIDTH : {ri.handbag_width_in_rf}')

    result_bench = base_model.predict(source='ReferenceImages/bench.jpeg')
    for j,box in enumerate(result_bench[0].boxes):
        if box.cls.item() == 13.0 :
            ri.bench_width_in_rf = box.xywh[0][2]
            print(f'-----------Bench RF Class Name ID : {box.cls.item()} - WIDTH : {ri.bench_width_in_rf}')

    result_couch = base_model.predict(source='ReferenceImages/couch.jpeg')
    for j,box in enumerate(result_couch[0].boxes):
        if box.cls.item() == 57.0 :
            ri.couch_width_in_rf = box.xywh[0][2]
            print(f'-----------Couch RF Class Name ID : {box.cls.item()} - WIDTH : {ri.couch_width_in_rf}')

    result_backpack = base_model.predict(source='ReferenceImages/backpack.jpeg')
    for j,box in enumerate(result_backpack[0].boxes):
        if box.cls.item() == 24.0 :
            ri.backpack_width_in_rf = box.xywh[0][2]
            print(f'-----------Backpack RF Class Name ID : {box.cls.item()} - WIDTH : {ri.backpack_width_in_rf}')


    result_laptop = base_model.predict(source='ReferenceImages/laptop.jpeg')
    for j,box in enumerate(result_laptop[0].boxes):
        if box.cls.item() == 63.0 :
            ri.laptop_width_in_rf = box.xywh[0][2]
            print(f'-----------Laptop RF Class Name ID : {box.cls.item()} - WIDTH : {ri.laptop_width_in_rf}')

    result_potted_plant = base_model.predict(source='ReferenceImages/pottedplant.jpeg')
    for j,box in enumerate(result_potted_plant[0].boxes):
        if box.cls.item() == 58.0 :
            ri.potted_plant_width_in_rf = box.xywh[0][2]
            print(f'-----------Potted Plant RF Class Name ID : {box.cls.item()} - WIDTH : {ri.potted_plant_width_in_rf}')

    result_TV = base_model.predict(source='ReferenceImages/TV.jpeg')
    for j,box in enumerate(result_TV[0].boxes):
        if box.cls.item() == 62.0 :
            ri.tv_width_in_rf = box.xywh[0][2]
            print(f'-----------TV RF Class Name ID : {box.cls.item()} - WIDTH : {ri.tv_width_in_rf}')


    result_dining_table = base_model.predict(source='ReferenceImages/diningtable.jpeg')
    for j,box in enumerate(result_dining_table[0].boxes):
        if box.cls.item() == 60.0 :
            ri.dining_table_in_rf = box.xywh[0][2]
            print(f'-----------Dining Table RF Class Name ID : {box.cls.item()} - WIDTH : {ri.dining_table_in_rf}')


    result_suitcase = base_model.predict(source='ReferenceImages/suitcase.jpeg')
    for j,box in enumerate(result_suitcase[0].boxes):
        if box.cls.item() == 28.0 :
            ri.suitcase_width_in_rf = box.xywh[0][2]
            print(f'-----------Suitcase RF Class Name ID : {box.cls.item()} - WIDTH : {ri.suitcase_width_in_rf}')


def OutdoorDetectReferenceImages():

    result_bicycle = base_model.predict(source='ReferenceImages/bicycle.jpg')
    for j,box in enumerate(result_bicycle[0].boxes):
        if box.cls.item() == 1.0 :
            ri.bicycle_width_in_rf = box.xywh[0][2]
            print(f'-----------Bicycle RF Class Name ID : {box.cls.item()} - WIDTH : {ri.bicycle_width_in_rf}')

    result_car = base_model.predict(source='ReferenceImages/car.jpg')
    for j,box in enumerate(result_car[0].boxes):
        if box.cls.item() == 2.0 :
            ri.car_width_in_rf = box.xywh[0][2]
            print(f'-----------Car RF Class Name ID : {box.cls.item()} - WIDTH : {ri.car_width_in_rf}')

    result_motorcycle = base_model.predict(source='ReferenceImages/motorcycle.jpeg')
    for j,box in enumerate(result_motorcycle[0].boxes):
        if box.cls.item() == 3.0 :
            ri.motorcycle_width_in_rf = box.xywh[0][2]
            print(f'-----------MotorCycle RF Class Name ID : {box.cls.item()} - WIDTH : {ri.motorcycle_width_in_rf}')

    result_stop_sign = base_model.predict(source='ReferenceImages/stopsign.jpg')
    for j,box in enumerate(result_stop_sign[0].boxes):
        if box.cls.item() == 11.0 :
            ri.stopsign_width_in_rf = box.xywh[0][2]
            print(f'-----------Stop Sign RF Class Name ID : {box.cls.item()} - WIDTH : {ri.stopsign_width_in_rf}')

    result_traffic_light = base_model.predict(source='ReferenceImages/trafficlight.jpeg')
    for j,box in enumerate(result_traffic_light[0].boxes):
        if box.cls.item() == 9.0 :
            ri.trafficlight_width_in_rf = box.xywh[0][2]
            print(f'-----------Traffic Light RF Class Name ID : {box.cls.item()} - WIDTH : {ri.trafficlight_width_in_rf}')

    result_parking_meter = base_model.predict(source='ReferenceImages/parkingmeter.JPG')
    for j,box in enumerate(result_parking_meter[0].boxes):
        if box.cls.item() == 12.0 :
            ri.parkingmeter_width_in_rf = box.xywh[0][2]
            print(f'-----------Parking Meter RF Class Name ID : {box.cls.item()} - WIDTH : {ri.parkingmeter_width_in_rf}')



def focal_length_finder (measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width

    return focal_length

# distance finder function
def distance_finder(focal_length, real_object_width, width_in_frmae):
    distance = (real_object_width * focal_length) / width_in_frmae
    return distance

def check_folders():
    paths = {
        'data_path' : 'data',
        'images_path' : 'data/images',
        'videos_path' : 'data/videos'

    }
    # Check whether the specified path exists or not
    notExist = list(({file_type: path for (file_type, path) in paths.items() if not os.path.exists(path)}).values())

    if notExist:
        print(f'Folder {notExist} does not exist. We will created')
        # Create a new directory because it does not exist
        for folder in notExist:
            os.makedirs(folder)
            print(f"The new directory {folder} is created!")


check_folders()
st.set_page_config(page_title="YOLO App", page_icon= "👀")
st.title('Virtual Vision')

source = ("Image", "Video","Door_Classification")
source_index = st.sidebar.selectbox("Select Input type", range(
    len(source)), format_func=lambda x: source[x])
start_yolo = st.button('Detect and Convert to Speech')
stop_yolo = st.button('Stop')

if source_index == 0:
    uploaded_file = st.sidebar.file_uploader(
        "Load File", type=['png', 'jpeg', 'jpg'])
    if uploaded_file is not None:
        is_valid = True
        with st.spinner(text='Loading...'):
            st.sidebar.image(uploaded_file)
            picture = Image.open(uploaded_file)
            picture = picture.save(f'data/images/{uploaded_file.name}')
            img_source = f'data/images/{uploaded_file.name}'

            output_image = f'runs/detect/predict/{uploaded_file.name}'

    else:
        is_valid = False
if source_index ==1:
    uploaded_file = st.sidebar.file_uploader("Upload Video", type=['mov','mp4'])
    if uploaded_file is not None:
        is_valid = True
        with st.spinner(text='Loading...'):
            st.sidebar.video(uploaded_file)
            with open(os.path.join("data", "videos", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
            video_source = f'data/videos/{uploaded_file.name}'
    else:
        is_valid = False

if source_index ==2:
    uploaded_file = st.sidebar.file_uploader(
        "Load File", type=['png', 'jpeg', 'jpg'])
    if uploaded_file is not None:
        is_valid = True
        with st.spinner(text='Loading...'):
            st.sidebar.image(uploaded_file)
            picture = Image.open(uploaded_file)
            picture = picture.save(f'data/images/{uploaded_file.name}')
            door_img_source = f'data/images/{uploaded_file.name}'

            door_output_image = f'runs/detect/predict/{uploaded_file.name}'

    else:
        is_valid = False
def detect_uploaded_video(source):
    logging.warning ('----------START detect uploaded video------------------')

    IndoorDetectReferenceImages()
    results = base_model.predict(source = source, save = True, imgsz=320, conf=0.5)

    size = len(results)
    index = 0
    distance=''
    distances = {}

    while index < size:
        # loop frame by frame by skipping 35 frames
        for c in results[index].boxes.cls.unique():
            if c in selected_detected_class:
                n = (results[index].boxes.cls == c).sum()  # detections per class

        for j,box in enumerate(results[index].boxes):
            width = box.xywh[0][2]


            c = box.cls.item()
            if box.cls.item() == 0 : #if the box is Person
                focal_person = focal_length_finder(KNOWN_DISTANCE, person_WIDTH, rf.person_width_in_rf)
                distance = distance_finder(focal_person, person_WIDTH, width)
                keyname = f'keyname-{index}-{j}'
                distances[keyname] = distance

        if distances:
            print('------------NEAREST OBJECT distance------'+ str(min(distances.values())))
            #get nearest object name distance from dictionary
            nearest_object_name = min(distances, key=distances.get)
            nearest_object_distance = distances.get(nearest_object_name)

            cloud_file = open('cloudspeech.txt','a+')
            if index == 0 :
                cloud_file.write(f'\n Person in {nearest_object_distance:.1f} meter')
            else:
                cloud_file.write(f'\n{nearest_object_distance:.1f}')
            cloud_file.close()
        index = index + 30

    logging.warning ('----------END detect uploaded video ------------------')

def detect_uploaded_photo(source):
    logging.warning ('----------START detect uploaded photo------------------')

    IndoorDetectReferenceImages()
    results = base_model.predict(source = source, save = True, imgsz=320, conf=0.5 )

    size = len(results)
    index = 0

    while index < size:
        cloud_file = open('cloudspeech.txt','a+')

        for c in results[index].boxes.cls.unique():

            n = (results[index].boxes.cls == c).sum()  # detections per class
            total_object_text = f"{n} {base_model.names[int(c)]}{'s' * (n > 1)}, "
            cloud_file.write(f'\n{total_object_text}')

        cloud_file.close()
        index = index + 1

    logging.warning ('----------END detect uploaded photo ------------------')

def detect_uploaded_door_photo(source):
    logging.warning ('----------START detect uploaded photo------------------')

    # IndoorDetectReferenceImages()
    results = custom_model.predict(source = source, save = True, imgsz=320, conf=0.5 )

    size = len(results)
    index = 0

    while index < size:
        cloud_file = open('cloudspeech.txt','a+')

        for c in results[index].boxes.cls.unique():

            n = (results[index].boxes.cls == c).sum()  # detections per class
            total_object_text = f"{n} {base_model.names[int(c)]}{'s' * (n > 1)}, "
            cloud_file.write(f'\n{total_object_text}')

        cloud_file.close()
        index = index + 1

    logging.warning ('----------END detect uploaded photo ------------------')
def read_textfile():

    logging.warning(f'--------------START Reading File ----------')

    file = open('cloudspeech.txt','r')
    speech_text = file.read().strip()
    file.close()

    logging.warning(f'--------------END Reading File ----------')

    return speech_text

def clear_text():

    logging.info('------------Clearing Text----------------')

    file = open('cloudspeech.txt','w')
    file.write('')
    file.close()

def generate_audio(text):
    audio_file = BytesIO()
    if text != '':
        tts = gTTS(f"{text}", lang='en')
        tts.write_to_fp(audio_file)

    return audio_file

if is_valid:

    if start_yolo:

        clear_text()

        if source_index == 0:
            with st.spinner(text='Audio loading...'):
                logging.warning('-----------------yolo image prediction start---------------------')

                detect_uploaded_photo(img_source)
                text = read_textfile()

                col1, col2 = st.columns(2)
                with col1:
                    st.image(img_source, caption="Selected Image")
                with col2:
                    st.image(output_image, caption="Model prediction")

                if text != '':
                    st.audio(generate_audio(text))
        if source_index ==1:
            with st.spinner(text='Audio loading...'):
                logging.warning('-----------------yolo video prediction start---------------------')

                detect_uploaded_video(video_source)
                text = read_textfile()

                if text != '':
                    st.audio(generate_audio(text))

            logging.warning ('-----------------------Audio END-----------------------------')
        if source_index == 2:
            with st.spinner(text='Audio loading...'):
                logging.warning('-----------------yolo image prediction start---------------------')

                detect_uploaded_door_photo(door_img_source)
                text = read_textfile()

                col1, col2 = st.columns(2)
                with col1:
                    st.image(door_img_source, caption="Selected Image")
                with col2:
                    st.image(door_output_image, caption="Model prediction")

                if text != '':
                    st.audio(generate_audio(text))

if stop_yolo and processes:
    #stop_process(*processes)
    processes.clear()
