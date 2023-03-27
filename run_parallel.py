from multiprocessing import Process
from ultralytics import YOLO
from gtts import gTTS
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import pyttsx3
import time
import ReferenceImageVal as ri
import signal
import streamlit as st
import numpy as np

model = YOLO('yolov8n.pt')  #yolov8n.pt load a pretrained model (recommended for training)


def DetectReferenceImages():
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



def Detect():

    DetectReferenceImages()
    model.predict(source="0", show = True)

def Speech():
    engine = pyttsx3.init()
    while True:
        with open('speech.txt') as f:
            speech = f.readlines()

        f.close()
        newVoiceRate = 170
        engine.setProperty('rate',newVoiceRate)

        if speech != []:
            engine.say(f'{speech}')
            engine.runAndWait()
        time.sleep(3)


def SpeechStreamLit():
    audio_file = open('myaudio.mp3', 'rb')
    audio_bytes = audio_file.read()

    st.audio(audio_bytes, format='audio/mp3')

    sample_rate = 44100  # 44100 samples per second
    seconds = 2  # Note duration of 2 seconds
    frequency_la = 440  # Our played note will be 440 Hz
    # Generate array with seconds*sample_rate steps, ranging between 0 and seconds
    t = np.linspace(0, seconds, seconds * sample_rate, False)
    # Generate a 440 Hz sine wave
    note_la = np.sin(frequency_la * t * 2 * np.pi)

    st.audio(note_la, sample_rate=sample_rate)


p1 = Process(target= Detect)
p2 = Process(target= Speech)

def stopProcess():
    p1.kill()
    p2.kill()
    print('--------------EXIT START------------------')
    signal.SIGINT
    print('--------------EXIT END------------------')

if __name__ == '__main__':

  p1.start()
  p2.start()
  #time.sleep(30)
  #stopProcess()
 # p1.join()
 # p2.join()
