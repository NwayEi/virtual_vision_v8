from multiprocessing import Process
from ultralytics import YOLO
from gtts import gTTS
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import pyttsx3
import time
import ReferenceImageVal as ri
import signal


model = YOLO('yolov8n.pt')  #yolov8n.pt load a pretrained model (recommended for training)


def DetectReferenceImages():
    #for i in range(80):
    result = model.predict(source='ReferenceImages/person.png')
    ri.person_width_in_rf = result[0].boxes.xywh[0][2]
    print(f'-----------Person width : {ri.person_width_in_rf}')

    result_cellphone= model.predict(source='ReferenceImages/cellphone.png')
    ri.mobile_width_in_rf = result_cellphone[0].boxes.xywh[0][2]
    print(f'-----------Cellphone width : {ri.mobile_width_in_rf}')

    result_handbag = model.predict(source='ReferenceImages/handbag.jpeg')
    ri.handbag_width_in_rf = result_handbag[0].boxes.xywh[0][2]
    print(f'-----------Handbag width : {ri.handbag_width_in_rf}')

    result_mouse = model.predict(source='ReferenceImages/mouse.jpeg')
    ri.mouse_width_in_rf = result_mouse[0].boxes.xywh[0][2]
    print(f'-----------Mouse width : {ri.mouse_width_in_rf}')


def Detect():

    result = model.predict(source='ReferenceImages/person.png')
    ri.person_width_in_rf = result[0].boxes.xywh[0][2]
    print(f'-----------Person width : {ri.person_width_in_rf}')

    result_cellphone= model.predict(source='ReferenceImages/cellphone.png')
    ri.mobile_width_in_rf = result_cellphone[0].boxes.xywh[0][2]
    print(f'-----------Cellphone width : {ri.mobile_width_in_rf}')

    result_handbag = model.predict(source='ReferenceImages/handbag.jpeg')
    ri.handbag_width_in_rf = result_handbag[0].boxes.xywh[0][2]
    print(f'-----------Handbag width : {ri.handbag_width_in_rf}')

    result_mouse = model.predict(source='ReferenceImages/mouse.jpeg')
    ri.mouse_width_in_rf = result_mouse[0].boxes.xywh[0][2]
    print(f'-----------Mouse width : {ri.mouse_width_in_rf}')

    result_bottle = model.predict(source='ReferenceImages/bottle.jpeg')
    ri.bottle_width_in_rf = result_bottle[0].boxes.xywh[0][2]
    print(f'-----------Bottle width : {ri.bottle_width_in_rf}')

    result_backpack = model.predict(source='ReferenceImages/backpack.jpeg')
    ri.backpack_width_in_rf = result_backpack[0].boxes.xywh[0][2]
    print(f'-----------Backpack width : {ri.backpack_width_in_rf}')

    result_laptop = model.predict(source='ReferenceImages/laptop.jpeg')
    ri.laptop_width_in_rf = result_laptop[0].boxes.xywh[0][2]
    print(f'-----------Laptop width : {ri.laptop_width_in_rf}')

    model.predict(source="0", show = True)

def Speech():
    engine = pyttsx3.init()
    while True:

        with open('/Users/orchidaung/speech.txt') as f:
            speech = f.readlines()

        f.close()
        newVoiceRate = 170
        engine.setProperty('rate',newVoiceRate)
        engine.say(f'{speech}')
        engine.runAndWait()
        time.sleep(3)



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
