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

def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
  lw = max(round(sum(image.shape) / 2 * 0.003), 2)
  p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
  cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
  if label:
    tf = max(lw - 1, 1)  # font thickness
    w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
    outside = p1[1] - h >= 3
    p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
    cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(image,
                label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0,
                lw / 3,
                txt_color,
                thickness=tf,
                lineType=cv2.LINE_AA)

def plot_bboxes(image, boxes, labels=[], colors=[], score=True, conf=None):
  #Define COCO Labels
    if labels == []:
        labels = {0: u'__background__', 1: u'person', 2: u'bicycle',3: u'car', 4: u'motorcycle', 5: u'airplane', 6: u'bus', 7: u'train', 8: u'truck', 9: u'boat', 10: u'traffic light', 11: u'fire hydrant', 12: u'stop sign', 13: u'parking meter', 14: u'bench', 15: u'bird', 16: u'cat', 17: u'dog', 18: u'horse', 19: u'sheep', 20: u'cow', 21: u'elephant', 22: u'bear', 23: u'zebra', 24: u'giraffe', 25: u'backpack', 26: u'umbrella', 27: u'handbag', 28: u'tie', 29: u'suitcase', 30: u'frisbee', 31: u'skis', 32: u'snowboard', 33: u'sports ball', 34: u'kite', 35: u'baseball bat', 36: u'baseball glove', 37: u'skateboard', 38: u'surfboard', 39: u'tennis racket', 40: u'bottle', 41: u'wine glass', 42: u'cup', 43: u'fork', 44: u'knife', 45: u'spoon', 46: u'bowl', 47: u'banana', 48: u'apple', 49: u'sandwich', 50: u'orange', 51: u'broccoli', 52: u'carrot', 53: u'hot dog', 54: u'pizza', 55: u'donut', 56: u'cake', 57: u'chair', 58: u'couch', 59: u'potted plant', 60: u'bed', 61: u'dining table', 62: u'toilet', 63: u'tv', 64: u'laptop', 65: u'mouse', 66: u'remote', 67: u'keyboard', 68: u'cell phone', 69: u'microwave', 70: u'oven', 71: u'toaster', 72: u'sink', 73: u'refrigerator', 74: u'book', 75: u'clock', 76: u'vase', 77: u'scissors', 78: u'teddy bear', 79: u'hair drier', 80: u'toothbrush'}
    #Define colors
    if colors == []:
    #colors = [(6, 112, 83), (253, 246, 160), (40, 132, 70), (205, 97, 162), (149, 196, 30), (106, 19, 161), (127, 175, 225), (115, 133, 176), (83, 156, 8), (182, 29, 77), (180, 11, 251), (31, 12, 123), (23, 6, 115), (167, 34, 31), (176, 216, 69), (110, 229, 222), (72, 183, 159), (90, 168, 209), (195, 4, 209), (135, 236, 21), (62, 209, 199), (87, 1, 70), (75, 40, 168), (121, 90, 126), (11, 86, 86), (40, 218, 53), (234, 76, 20), (129, 174, 192), (13, 18, 254), (45, 183, 149), (77, 234, 120), (182, 83, 207), (172, 138, 252), (201, 7, 159), (147, 240, 17), (134, 19, 233), (202, 61, 206), (177, 253, 26), (10, 139, 17), (130, 148, 106), (174, 197, 128), (106, 59, 168), (124, 180, 83), (78, 169, 4), (26, 79, 176), (185, 149, 150), (165, 253, 206), (220, 87, 0), (72, 22, 226), (64, 174, 4), (245, 131, 96), (35, 217, 142), (89, 86, 32), (80, 56, 196), (222, 136, 159), (145, 6, 219), (143, 132, 162), (175, 97, 221), (72, 3, 79), (196, 184, 237), (18, 210, 116), (8, 185, 81), (99, 181, 254), (9, 127, 123), (140, 94, 215), (39, 229, 121), (230, 51, 96), (84, 225, 33), (218, 202, 139), (129, 223, 182), (167, 46, 157), (15, 252, 5), (128, 103, 203), (197, 223, 199), (19, 238, 181), (64, 142, 167), (12, 203, 242), (69, 21, 41), (177, 184, 2), (35, 97, 56), (241, 22, 161)]
        colors = [(89, 161, 197),(67, 161, 255),(19, 222, 24),(186, 55, 2),(167, 146, 11),(190, 76, 98),(130, 172, 179),(115, 209, 128),(204, 79, 135),(136, 126, 185),(209, 213, 45),(44, 52, 10),(101, 158, 121),(179, 124, 12),(25, 33, 189),(45, 115, 11),(73, 197, 184),(62, 225, 221),(32, 46, 52),(20, 165, 16),(54, 15, 57),(12, 150, 9),(10, 46, 99),(94, 89, 46),(48, 37, 106),(42, 10, 96),(7, 164, 128),(98, 213, 120),(40, 5, 219),(54, 25, 150),(251, 74, 172),(0, 236, 196),(21, 104, 190),(226, 74, 232),(120, 67, 25),(191, 106, 197),(8, 15, 134),(21, 2, 1),(142, 63, 109),(133, 148, 146),(187, 77, 253),(155, 22, 122),(218, 130, 77),(164, 102, 79),(43, 152, 125),(185, 124, 151),(95, 159, 238),(128, 89, 85),(228, 6, 60),(6, 41, 210),(11, 1, 133),(30, 96, 58),(230, 136, 109),(126, 45, 174),(164, 63, 165),(32, 111, 29),(232, 40, 70),(55, 31, 198),(148, 211, 129),(10, 186, 211),(181, 201, 94),(55, 35, 92),(129, 140, 233),(70, 250, 116),(61, 209, 152),(216, 21, 138),(100, 0, 176),(3, 42, 70),(151, 13, 44),(216, 102, 88),(125, 216, 93),(171, 236, 47),(253, 127, 103),(205, 137, 244),(193, 137, 224),(36, 152, 214),(17, 50, 238),(154, 165, 67),(114, 129, 60),(119, 24, 48),(73, 8, 110)]

    #plot each boxes
    for box in boxes:
    #add score in label if score=True
        if score :
            label = labels[int(box[-1])+1] + " " + str(round(100 * float(box[-2]),1)) + "%"
        else :
            label = labels[int(box[-1])+1]
        #filter every box under conf threshold if conf threshold setted
        if conf :
            if box[-2] > conf:
                color = colors[int(box[-1])]
                box_label(image, box, label, color)
        else:
            color = colors[int(box[-1])]
            box_label(image, box, label, color)

  #show image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow(image)

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
