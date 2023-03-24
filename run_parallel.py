import multiprocessing as mp
from multiprocessing import Process
from ultralytics import YOLO
# from gtts import gTTS
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import pyttsx3
import time
import subprocess
import streamlit as st
import signal
import os



stop_event = mp.Event()
start_yolo = st.button("Start")
stop_yolo = st.button("Stop")
running = False
processes =[]
pid = None


def Detect(stop_event):
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
    while not stop_event.is_set():
        results = model.predict(source="0", show=True)

    print('---------------------detect------------')

def Speech(stop_event):
    engine = pyttsx3.init()
    with open('/Users/thm/code/NwayEi/virtual_vision/virtual_vision_v8/speech.txt') as f:
        speech = f.readlines()
    f.close()
    newVoiceRate = 170
    engine.setProperty('rate',newVoiceRate)
    while not stop_event.is_set():
        engine.say(f'{speech}')
        engine.runAndWait()
        time.sleep(3)
    print('------------------------speech------------')


def run():
    global running
    stop_event.clear()

    p1 = Process(target=Detect, args=(stop_event,))
    p1.start()
    processes.append(p1)

    p2 = Process(target=Speech, args=(stop_event,))
    p2.start()
    processes.append(p2)

    p1.join()
    p2.join()

    running = True

    st.session_state['pid'] = [p.pid for p in processes]


def stop_run():
    global running, processes
    if running:
        stop_event.set()
        for p in processes:
            p.terminate()
        print("YOLO and speech processes have stopped.")
        running = False
        st.session_state['pid'] = None

        st.write("YOLO + Speech stopped.")
    else:
        st.write("YOLO + Speech is not running.")


    print('--------------------------------stop------------')

# if __name__ == '__main__':
if start_yolo:
    run()
elif stop_yolo:
    stop_run()
