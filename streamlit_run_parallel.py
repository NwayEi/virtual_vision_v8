from multiprocessing import Process
from ultralytics import YOLO
from gtts import gTTS
import ReferenceImageVal as ri
import signal
import streamlit as st
import numpy as np
import base64
import logging
import cv2
from component import FrameCounter
import av
import queue
import subprocess
import threading

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# st.set_page_config( layout="wide" )
# logger = logging.getLogger( __name__ )
video_stream= None
model = YOLO('yolov8n.pt')  #yolov8n.pt load a pretrained model (recommended for training)
# start_run = st.button('start_yolo')
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)





st.set_page_config(page_title="Streamlit WebRTC Demo", page_icon="🤖")
task_list = ["Video Stream"]

with st.sidebar:
    st.title('Task Selection')
    task_name = st.selectbox("Select your tasks:", task_list)
st.title(task_name)

if task_name == task_list[0]:

    style_list = ['color', 'black and white']

    st.sidebar.header('Style Selection')
    style_selection = st.sidebar.selectbox("Choose your style:", style_list)


    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.model_lock = threading.Lock()
            self.model = YOLO('yolov8n.pt')

        def detect_objects(self, img):
            # # Resize image to match the input size of the YOLO model
            # img = cv2.resize(img, (416, 416))

            # Convert image to input format expected by the YOLO model
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            # Run object detection using the YOLO model
            preds = self.model.predict(img)
        def recv(self, frame):
            preds = frame.to_ndarray(format="bgr24")
            output_img = self.detect_objects(preds)
            return av.VideoFrame.from_ndarray(output_img, format="bgr24")
    ctx = webrtc_streamer(
        key="example",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": True,
            "audio": False})
    # if ctx.video_processor:
    #     print('--------video start----------------')
    #     video_stream = ctx.video_transformer.update_style(style_selection)

#-----------------------this code works for video streaming but not Yolo -------------------------
#     class VideoProcessor(VideoProcessorBase):
#         def __init__(self):
#             self.model_lock = threading.Lock()
#             self.style = style_list[0]

#         def update_style(self, new_style):
#             if self.style != new_style:
#                 with self.model_lock:
#                     self.style = new_style

#         def recv(self, frame):
#             # img = frame.to_ndarray(format="bgr24")
#             img = frame.to_image()
#             if self.style == style_list[1]:
#                 img = img.convert("L")

#             # return av.VideoFrame.from_ndarray(img, format="bgr24")
#             return av.VideoFrame.from_image(img)

#     ctx = webrtc_streamer(
#         key="example",
#         video_processor_factory=VideoProcessor,
#         rtc_configuration=RTC_CONFIGURATION,
#         media_stream_constraints={
#             "video": True,
#             "audio": False}
# )

#     if ctx.video_processor:
#         print('--------video start----------------')
#         video_stream = ctx.video_transformer.update_style(style_selection)

#----------------------------------this code ends for video streaming ----------------------------
        # detectreferenceimages()
        # print('-------------------detect start------------------------')
        # model.predict(source=video_stream, show = )
        # print('-------------------predict start------------------------')

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
# def live_object_detection():

#     # weight, confidence_threshold, track_age, track_hits, iou_thres = variables.get_var()

#       # init frame counter, object detector, tracker and passing object counter
#     frame_num = FrameCounter()
#     detector = detect()
#     print ('-----------------------------in midst of running detection----------------------')
#     if 'counters' not in st.session_state:
#         st.session_state.counters = []
#     icounter = st.session_state.counters

#     # Dump queue for real time detection result
#     # result_queue = (queue.Queue())
#     # frame_queue = (queue.Queue( maxsize=1 ))

#     # reading each frame of live stream and passing to backend processing
#     def frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
#         frame = frame.to_ndarray( format="bgr24" )

#         # # Detect, track and counter the intersect of objects here
#         # image, result = deepsort_tracker.track_video_stream( frame, frame_num( 1 ) )
#         # if icounter is not None:
#         #     if len( icounter ) > 0:
#         #         image = st_icounter.update_counters( deepsort_tracker.tracker.tracks, image, icounter )

#         # NOTE: This `recv` method is called in another thread,
#         # so it must be thread-safe.
#         # result_queue.put( result )
#         # if not frame_queue.full():
#         #     frame_queue.put( frame )

#         return av.VideoFrame.from_ndarray(frame, format="bgr24" )
#         print ('--------------------------------is this needed?-------------------------------------')

#     RTC_CONFIGURATION = RTCConfiguration( {"iceServers": servers} )
#     webrtc_ctx = webrtc_streamer(
#         key="object-detection",
#         mode=WebRtcMode.SENDRECV,
#         rtc_configuration=RTC_CONFIGURATION,  # when deploy on remote host need stun server for camera connection
#         video_frame_callback=frame_callback,
#         media_stream_constraints={"video": True, "audio": False},
#         async_processing=True,
#         )

#     # capture image for the counter setup container
#     if webrtc_ctx.state.playing:
#         # image = frame_queue.get()
#         # st_icounter = ic.st_IntersectCounter( image, image.shape[1], image.shape[0] )
#         icounter = st.session_state.counters
#         if len( st.session_state.counters ) > 0:
#             st.session_state.counted = True
#         labels_placeholder = st.empty()

#         # NOTE: The video transformation with object detection and
#         # this loop displaying the result labels are running
#         # in different threads asynchronously.
#         # Then the rendered video frames and the labels displayed here
#         # are not strictly synchronized.
#         # while True:
#         #     try:
#         #         result = result_queue.get( timeout=1.0 )
#         #         labels_placeholder.dataframe( session_result.result_to_df( result ), use_container_width=True )
#         #     except queue.Empty:
#         #         result = None
#             # if st_icounter is not None:
#             #     st_icounter.show_counter_results()

#     else:
#         st.session_state.counters = []
#         st.session_state.counters_table = []
#         st.session_state.counted = False
#         st.session_state.result_list = []





    # results = subprocess.run(['yolo', 'task=detect', 'mode=predict', 'model=yolov8n.pt', 'conf=0.25', 'source=0'],capture_output=True, universal_newlines=True).stderr


# if start_run:
#     live_object_detection()

#     print('---------------start running--------------------')
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

# def autoplay_audio(file_path: str):
#     with open(file_path, "rb") as f:
#         data = f.read()
#         b64 = base64.b64encode(data).decode()
#         md = f"""
#             <audio controls autoplay="true">
#             <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
#             </audio>
#             """
#         st.markdown(
#             md,
#             unsafe_allow_html=True,
#         ).write("# Auto-playing Audio!")

# # #autoplay_audio("local_audio.mp3")

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


# if __name__ == "__main__":


#     # DEBUG = config.DEBUG

#     # if DEBUG:
#     #     logging.basicConfig(
#     #         format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
#     #                "%(message)s",
#     #         force=True,
#     #     )

#     #     logger.setLevel( level=logging.DEBUG if DEBUG else logging.INFO )

#     #     st_webrtc_logger = logging.getLogger( "streamlit_webrtc" )
#     #     st_webrtc_logger.setLevel( logging.DEBUG )

#     #     fsevents_logger = logging.getLogger( "fsevents" )
#     #     fsevents_logger.setLevel( logging.WARNING )

#     main()
