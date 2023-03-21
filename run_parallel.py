from multiprocessing import Process
from ultralytics import YOLO
from gtts import gTTS
from collections import Counter
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import pyttsx3
import time

def Detect():
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
    results = model.predict(source="0", show = True)

def Speech():
    engine = pyttsx3.init()
    while True:

        with open('/Users/orchidaung/speech.txt') as f:
            speech = f.readlines()

        f.close()
        print(f'{speech}')
        engine.say(f'{speech}')
        engine.runAndWait()
        time.sleep(3)


if __name__ == '__main__':
  p1 = Process(target=Detect)
  p1.start()
  p2 = Process(target=Speech)
  p2.start()
  p1.join()
  p2.join()
