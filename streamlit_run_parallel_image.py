import multiprocessing as mp
from multiprocessing import Process
from ultralytics import YOLO
from gtts import gTTS
import ReferenceImageVal as ri
import signal
import streamlit as st
import numpy as np
import base64
import subprocess
import os
import pyttsx3
import time
import threading
import logging
import pyttsx3
import time
from io import BytesIO
from PIL import Image
import speech_recognition as sr
import cv2
import torch
from pathlib import Path
import PIL
from ultralytics.yolo.utils.plotting import Annotator
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import matplotlib.pyplot as plt
processes=[]
model = YOLO('yolov8n.pt')  #yolov8n.pt load a pretrained model (recommended for training)
# is_gtts = False
#-------------------------------------------------v1 for streamlit--------------------------------------------------------

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
st.title('YOLO App for Video and Image')

source = ("Image", "Video")
source_index = st.sidebar.selectbox("Select Input type", range(
    len(source)), format_func=lambda x: source[x])
start_yolo = st.button('Detect')
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

    else:
        is_valid = False
else:
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


def detectreferenceimages():
    result = model.predict(source='ReferenceImages/person.png')
    ri.person_width_in_rf = result[0].boxes.xywh[0][2]
    print(f'-----------Person width : {ri.person_width_in_rf}')

    result_chair= model.predict(source='ReferenceImages/chair.jpeg')
    ri.chair_width_in_rf = result_chair[0].boxes.xywh[0][2]
    print(f'-----------Chair width : {ri.chair_width_in_rf}')

    result_handbag = model.predict(source='ReferenceImages/handbag.jpeg')
    ri.handbag_width_in_rf = result_handbag[0].boxes.xywh[0][2]
    print(f'-----------Handbag width : {ri.handbag_width_in_rf}')

    result_bench = model.predict(source='ReferenceImages/bench.jpeg')
    ri.bench_width_in_rf = result_bench[0].boxes.xywh[0][2]
    print(f'-----------Bench width : {ri.bench_width_in_rf}')

    result_couch = model.predict(source='ReferenceImages/couch.jpeg')
    ri.couch_width_in_rf = result_couch[0].boxes.xywh[0][2]
    print(f'-----------Couch width : {ri.couch_width_in_rf}')

    result_backpack = model.predict(source='ReferenceImages/backpack.jpeg')
    ri.backpack_width_in_rf = result_backpack[0].boxes.xywh[0][2]
    print(f'-----------Backpack width : {ri.backpack_width_in_rf}')

    result_laptop = model.predict(source='ReferenceImages/laptop.jpeg')
    ri.laptop_width_in_rf = result_laptop[0].boxes.xywh[0][2]
    print(f'-----------Laptop width : {ri.laptop_width_in_rf}')


def detect_uploaded(source):
    logging.warning ('----------START model  prediction------------------')
    results = model.predict(source = source)
    logging.warning ('----------END model  prediction ------------------')

# def speech_uploaded_video():
#     logging.warning ('----------speech start------------------')
#     engine = pyttsx3.init()
#     newVoiceRate = 170
#     engine.setProperty('rate', newVoiceRate)

#     while True:
#         logging.warning(f'--------------START Reading File ----------')
#         file = open('speech.txt','r')
#         speech_text = file.read().strip()
#         file.close()
#         logging.warning(f'--------------END Reading File ----------{speech_text}')


#         if speech_text != '':
#             text_to_speech(speech_text)
#             #engine.say(speech_text)
#             #engine.runAndWait()

#          #Process the result here...

#         time.sleep(3)
#         logging.warning ('----------speech end------------------')
def text_to_speech(text):

    logging.warning ('----------START Text to speech ------------------')
    #gtts = gTTS(text, lang='en')
    #gtts.save('testspeech.mp3')
    audio_file = open(f'myaudio.mp3','rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format ='audio/mp3', start_time = 0)
    logging.warning ('----------END Text to speech ------------------')
    logging.warning(f'--------------START Reading File ----------')
    file = open('speech.txt','r')
    speech_text = file.read().strip()
    file.close()

    logging.warning(f'--------------END Reading File ----------{speech_text}')

def speech_uploaded_video():

    logging.warning(f'--------------START Reading File ----------')
    file = open('speech.txt','r')
    speech_text = file.read().strip()
    file.close()

    logging.warning(f'--------------END Reading File ----------{speech_text}')

    return speech_text

def autoplay_audio(file_path: str):
     with open(file_path, "rb") as f:
         data = f.read()
         b64 = base64.b64encode(data).decode()
         md = f"""
             <audio autoplay>
             <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
             </audio>
             """
         st.markdown(
             md,
             unsafe_allow_html=True,
         ).write("# Auto-playing Audio!")



if is_valid:

    if start_yolo:

        if source_index == 0:
            # if st.sidebar.checkbox("Gender identification"):
            #         assigned_class = st.sidebar.multiselect("Select Classes", "Female", "Male")

            with st.spinner(text='Audio loading...'):
                logging.warning('-----------------yolo image prediction start---------------------')

                res = model.predict(img_source)
                output_image = f'runs/detect/train/{uploaded_file.name}'

                text = speech_uploaded_video()
                sound_file_img = BytesIO()
                tts = gTTS(f"{text}", lang='en')
                tts.write_to_fp(sound_file_img)
                st.audio(sound_file_img)


                col1, col2 = st.columns(2)
                with col1:
                    st.image(img_source, caption="Selected Image")
                with col2:
                    st.image(output_image, caption="Model prediction")


        else:
            with st.spinner(text='Audio loading...'):
                logging.warning('-----------------yolo video prediction start---------------------')

                detect_uploaded(video_source)
                text = speech_uploaded_video()

                sound_file_video = BytesIO()
                tts = gTTS(f"{text}", lang='en')
                tts.write_to_fp(sound_file_video)
                st.audio(sound_file_video)

            logging.warning ('-----------------------Audio END-----------------------------')

if stop_yolo and processes:
    #stop_process(*processes)
    processes.clear()



#--------------------------------------------------v1 for streamlit ended-------------------------------------------------------


# -----------------------------------v2 streamlit start-----------------------------------------
# cfg_model_path = "ultralytics/yolo/engine/models/'yolov8n.pt'"

# def videoinput():
#     uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'mpeg', 'mov'])
#     if uploaded_video != None:

#         imgpath = os.path.join('data/videos', uploaded_video.name)
#         outputpath = os.path.join('data/output', os.path.basename(imgpath))

#         with open(imgpath, mode='wb') as f:
#             f.write(uploaded_video.read())  # save video to disk

#         st_video = open(imgpath, 'rb')
#         video_bytes = st_video.read()
#         st.video(video_bytes)
#         st.write("Uploaded Video")
#         model.predict (source = imgpath)
#         st_video2 = open(outputpath, 'rb')
#         video_bytes2 = st_video2.read()
#         st.video(video_bytes2)
#         st.write("Model Prediction")

# def main():

#     option = st.sidebar.radio("Select input type.", ['Image', 'Video'])

#     st.header('Obstacle Detection')
#     st.subheader('Select options left-haned menu bar.')
#     # if option == "Image"
#     #     imageInput(deviceoption, datasrc)
#     if option == "Video":
#         videoinput()
# if __name__ == '__main__':
#     main()
#-----------------------------------v2 for streamlit end-----------------------------------#

# def detectreferenceimages(self):
#             #for i in range(80):
#             result = self.model.predict(source='ReferenceImages/person.png')
#             ri.person_width_in_rf = result[0].boxes.xywh[0][2]
#             print(f'-----------Person width : {ri.person_width_in_rf}')

#             result_cellphone= self.model.predict(source='ReferenceImages/cellphone.png')
#             ri.mobile_width_in_rf = result_cellphone[0].boxes.xywh[0][2]
#             print(f'-----------Cellphone width : {ri.mobile_width_in_rf}')

#             result_handbag = self.model.predict(source='ReferenceImages/handbag.jpeg')
#             ri.handbag_width_in_rf = result_handbag[0].boxes.xywh[0][2]
#             print(f'-----------Handbag width : {ri.handbag_width_in_rf}')

#             result_mouse = self.model.predict(source='ReferenceImages/mouse.jpeg')
#             ri.mouse_width_in_rf = result_mouse[0].boxes.xywh[0][2]
#             print(f'-----------Mouse width : {ri.mouse_width_in_rf}')
#



# p1 = Process(target= Detect)
# p2 = Process(target= autoplay_audio('myaudio.mp3'))

# def stopProcess():
#     p1.kill()
#     p2.kill()
#     print('--------------EXIT START------------------')
#     signal.SIGINT
#     print('--------------EXIT END------------------')

# if __name__ == '__main__':

#   p1.start()
#   p2.start()
#   #time.sleep(30)
#   #stopProcess()
#  # p1.join()
#  # p2.join()








#--------------------------------code for audio file retrieval ---------------------------------------------------------------

# def speechstreamlit():
#     audio_file = open('myaudio.mp3', 'rb')
#     audio_bytes = audio_file.read()

#     st.audio(audio_bytes, format='audio/mp3')

#     sample_rate = 44100  # 44100 samples per second
#     seconds = 2  # Note duration of 2 seconds
#     frequency_la = 440  # Our played note will be 440 Hz
#     # Generate array with seconds*sample_rate steps, ranging between 0 and seconds
#     t = np.linspace(0, seconds, seconds * sample_rate, False)
#     # Generate a 440 Hz sine wave
#     note_la = np.sin(frequency_la * t * 2 * np.pi)

#     st.audio(note_la, sample_rate=sample_rate)



# def speech():
#     print('---------------speech func started------------------')
#     engine = pyttsx3.init()

#     while True:

#         with open('speech.txt') as f:
#             speech = f.readlines()
#         f.close()
#         newVoiceRate = 170
#         engine.setProperty('rate',newVoiceRate)

#         if speech != []:
#             engine.say(f'{speech}')
#             engine.runAndWait()
#         time.sleep(3)

#         print('---------------speech func ended--------------------------')

#----------------------------end code for audio file retrieval ---------------------------------------------------------------



# #-----------------------this code works for video streaming but not Yolo -------------------------
# import threading

# from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration,VideoHTMLAttributes

# RTC_CONFIGURATION = RTCConfiguration(
#     {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
# )

# st.set_page_config(page_title="Streamlit WebRTC Demo", page_icon="🤖")
# task_list = ["Video Stream"]

# with st.sidebar:
#     st.title('Task Selection')
#     task_name = st.selectbox("Select your tasks:", task_list)
# st.title(task_name)

# if task_name == task_list[0]:

#     style_list = ['color', 'black and white']

#     st.sidebar.header('Style Selection')
#     style_selection = st.sidebar.selectbox("Choose your style:", style_list)

#     class VideoProcessor(VideoProcessorBase):
#         def __init__(self):
#             self.model_lock = threading.Lock()
#             self.style = style_list[0]

#         def update_style(self, new_style):
#             if self.style != new_style:
#                 with self.model_lock:
#                     self.style = new_style

#         def recv(self, frame):
#             img = frame.to_ndarray(format="bgr24")
#             img = frame.to_image()
#             if self.style == style_list[1]:
#                 img = img.convert("L")

#             # return av.VideoFrame.from_ndarray(img, format="bgr24")
#             return av.VideoFrame.from_image(img)

#     ctx = webrtc_streamer(
#         key= "object detection",
#         video_html_attrs= VideoHTMLAttributes(autoplay=True,controls=True),      video_processor_factory=VideoProcessor,
#         rtc_configuration=RTC_CONFIGURATION,
#         media_stream_constraints={
#             "video": True,
#             "audio": False}
# )

#     if ctx.video_processor:
#         print('--------video start----------------')
#         video_stream = ctx.video_transformer.update_style(style_selection)

#----------------------------------this code ends for video streaming ----------------------------
