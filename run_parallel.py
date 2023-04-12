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

def IndoorDetectReferenceImages():

    result_person = model.predict(source='ReferenceImages/person.png')
    for j,box in enumerate(result_person[0].boxes):
        if box.cls.item() == 0.0 :
            ri.person_width_in_rf = box.xywh[0][2]
            print(f'-----------Person RF Class Name ID : {box.cls.item()} - WIDTH : {ri.person_width_in_rf}')

    result_chair= model.predict(source='ReferenceImages/chair.jpeg')
    for j,box in enumerate(result_chair[0].boxes):
        if box.cls.item() == 56.0 :
            ri.chair_width_in_rf = box.xywh[0][2]
            print(f'-----------Chair RF Class Name ID : {box.cls.item()} - WIDTH : {ri.chair_width_in_rf}')

    result_handbag = model.predict(source='ReferenceImages/handbag.jpeg')
    for j,box in enumerate(result_handbag[0].boxes):
        if box.cls.item() == 26.0 :
            ri.handbag_width_in_rf = box.xywh[0][2]
            print(f'-----------HandBag RF Class Name ID : {box.cls.item()} - WIDTH : {ri.handbag_width_in_rf}')

    result_bench = model.predict(source='ReferenceImages/bench.jpeg')
    for j,box in enumerate(result_bench[0].boxes):
        if box.cls.item() == 13.0 :
            ri.bench_width_in_rf = box.xywh[0][2]
            print(f'-----------Bench RF Class Name ID : {box.cls.item()} - WIDTH : {ri.bench_width_in_rf}')

    result_couch = model.predict(source='ReferenceImages/couch.jpeg')
    for j,box in enumerate(result_couch[0].boxes):
        if box.cls.item() == 57.0 :
            ri.couch_width_in_rf = box.xywh[0][2]
            print(f'-----------Couch RF Class Name ID : {box.cls.item()} - WIDTH : {ri.couch_width_in_rf}')

    result_backpack = model.predict(source='ReferenceImages/backpack.jpeg')
    for j,box in enumerate(result_backpack[0].boxes):
        if box.cls.item() == 24.0 :
            ri.backpack_width_in_rf = box.xywh[0][2]
            print(f'-----------Backpack RF Class Name ID : {box.cls.item()} - WIDTH : {ri.backpack_width_in_rf}')


    result_laptop = model.predict(source='ReferenceImages/laptop.jpeg')
    for j,box in enumerate(result_laptop[0].boxes):
        if box.cls.item() == 63.0 :
            ri.laptop_width_in_rf = box.xywh[0][2]
            print(f'-----------Laptop RF Class Name ID : {box.cls.item()} - WIDTH : {ri.laptop_width_in_rf}')

    result_potted_plant = model.predict(source='ReferenceImages/pottedplant.jpeg')
    for j,box in enumerate(result_potted_plant[0].boxes):
        if box.cls.item() == 58.0 :
            ri.potted_plant_width_in_rf = box.xywh[0][2]
            print(f'-----------Potted Plant RF Class Name ID : {box.cls.item()} - WIDTH : {ri.potted_plant_width_in_rf}')

    result_TV = model.predict(source='ReferenceImages/TV.jpeg')
    for j,box in enumerate(result_TV[0].boxes):
        if box.cls.item() == 62.0 :
            ri.tv_width_in_rf = box.xywh[0][2]
            print(f'-----------TV RF Class Name ID : {box.cls.item()} - WIDTH : {ri.tv_width_in_rf}')


    result_dining_table = model.predict(source='ReferenceImages/diningtable.jpeg')
    for j,box in enumerate(result_dining_table[0].boxes):
        if box.cls.item() == 60.0 :
            ri.dining_table_in_rf = box.xywh[0][2]
            print(f'-----------Dining Table RF Class Name ID : {box.cls.item()} - WIDTH : {ri.dining_table_in_rf}')


    result_suitcase = model.predict(source='ReferenceImages/suitcase.jpeg')
    for j,box in enumerate(result_suitcase[0].boxes):
        if box.cls.item() == 28.0 :
            ri.suitcase_width_in_rf = box.xywh[0][2]
            print(f'-----------Suitcase RF Class Name ID : {box.cls.item()} - WIDTH : {ri.suitcase_width_in_rf}')

def OutdoorDetectReferenceImages():

    result_bicycle = model.predict(source='ReferenceImages/bicycle.jpg')
    for j,box in enumerate(result_bicycle[0].boxes):
        if box.cls.item() == 1.0 :
            ri.bicycle_width_in_rf = box.xywh[0][2]
            print(f'-----------Bicycle RF Class Name ID : {box.cls.item()} - WIDTH : {ri.bicycle_width_in_rf}')

    result_car = model.predict(source='ReferenceImages/car.jpg')
    for j,box in enumerate(result_car[0].boxes):
        if box.cls.item() == 2.0 :
            ri.car_width_in_rf = box.xywh[0][2]
            print(f'-----------Car RF Class Name ID : {box.cls.item()} - WIDTH : {ri.car_width_in_rf}')

    result_motorcycle = model.predict(source='ReferenceImages/motorcycle.jpeg')
    for j,box in enumerate(result_motorcycle[0].boxes):
        if box.cls.item() == 3.0 :
            ri.motorcycle_width_in_rf = box.xywh[0][2]
            print(f'-----------MotorCycle RF Class Name ID : {box.cls.item()} - WIDTH : {ri.motorcycle_width_in_rf}')

    result_stop_sign = model.predict(source='ReferenceImages/stopsign.jpg')
    for j,box in enumerate(result_stop_sign[0].boxes):
        if box.cls.item() == 11.0 :
            ri.stopsign_width_in_rf = box.xywh[0][2]
            print(f'-----------Stop Sign RF Class Name ID : {box.cls.item()} - WIDTH : {ri.stopsign_width_in_rf}')

    result_traffic_light = model.predict(source='ReferenceImages/trafficlight.jpeg')
    for j,box in enumerate(result_traffic_light[0].boxes):
        if box.cls.item() == 9.0 :
            ri.trafficlight_width_in_rf = box.xywh[0][2]
            print(f'-----------Traffic Light RF Class Name ID : {box.cls.item()} - WIDTH : {ri.trafficlight_width_in_rf}')

    result_parking_meter = model.predict(source='ReferenceImages/parkingmeter.JPG')
    for j,box in enumerate(result_parking_meter[0].boxes):
        if box.cls.item() == 12.0 :
            ri.parkingmeter_width_in_rf = box.xywh[0][2]
            print(f'-----------Parking Meter RF Class Name ID : {box.cls.item()} - WIDTH : {ri.parkingmeter_width_in_rf}')


def Detect():
    IndoorDetectReferenceImages()
    OutdoorDetectReferenceImages()
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

realtime_detection = Process(target= Detect)
audio_process = Process(target= Speech)

def stopProcess():
    realtime_detection.kill()
    audio_process.kill()
    print('--------------EXIT START------------------')
    signal.SIGINT
    print('--------------EXIT END------------------')

if __name__ == '__main__':

  realtime_detection.start()
  time.sleep(10)
  audio_process.start()
