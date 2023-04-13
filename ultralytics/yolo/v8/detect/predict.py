# Ultralytics YOLO 🚀, GPL-3.0 license

import torch

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from gtts import gTTS
import ReferenceImageVal as rf
new_text =''
old_text = ''
selected_detected_class = [0,13,26,56,24,57,63,58,62,60,28,1,2,3,9,11,12]


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


def focal_length_finder (measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width

    return focal_length

# distance finder function
def distance_finder(focal_length, real_object_width, width_in_frmae):
    distance = (real_object_width * focal_length) / width_in_frmae
    return distance

class DetectionPredictor(BasePredictor):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None):
        super().__init__(cfg, overrides)
        self.old_text = ''
        self.new_text = ''

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_imgs):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)
        print('---------------------POST PROCESS PREDICTION -----------')
        print(preds)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            path, _, _, _, _ = self.batch
            img_path = path[i] if isinstance(path, list) else path
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results



    def write_results(self, idx, results, batch):
        p, im, im0 = batch
        log_string = ''
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        imc = im0.copy() if self.args.save_crop else im0
        if self.source_type.webcam or self.source_type.from_img:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count

        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = results[idx].boxes  # TODO: make boxes inherit from tensors


        if len(det) == 0:
            file = open('speech.txt','w')
            a = file.write('')
            file.close()
            return f'{log_string}(no detections), '


        (H,W) = im.shape[:2]
        distance = 0

        for c in det.cls.unique():
            if c in selected_detected_class:
                n = (det.cls == c).sum()  # detections per class
                log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
                self.new_text += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

        distances = {}
        positions = {}

        for i, box in enumerate(reversed(det)):

            c = int(box.cls.item())
            confidenct_score = float(box.conf)
            name = self.model.names[int(c)]
            keyname = f'{name}-{i}'
            x, y, width, height = box.xywh[0]

            distance = 0
            if c == 56 and confidenct_score >= 0.4:
                if c == 56: # 4. Chair
                    focal_chair = focal_length_finder(KNOWN_DISTANCE, chair_WIDTH, rf.chair_width_in_rf )
                    distance = distance_finder(focal_chair, chair_WIDTH, width)
                    custom_distance = distance/3
                    distances[keyname] = custom_distance

            if confidenct_score >= 0.5:

                if c == 0 : # 1. Person
                    focal_person = focal_length_finder(KNOWN_DISTANCE, person_WIDTH, rf.person_width_in_rf)
                    distance = distance_finder(focal_person, person_WIDTH, width)
                    distances[keyname] = distance

                if c == 13: # 2. Bench
                    focal_bench = focal_length_finder(KNOWN_DISTANCE, bench_WIDTH, rf.bench_width_in_rf )
                    distance = distance_finder(focal_bench, bench_WIDTH, width)
                    distances[keyname] = distance

                if c == 26: # 3. Handbag
                    focal_handbag= focal_length_finder(KNOWN_DISTANCE, handbag_WIDTH, rf.handbag_width_in_rf )
                    distance = distance_finder(focal_handbag, handbag_WIDTH, width)
                    distances[keyname] = distance

                #if c == 56: # 4. Chair
                 #   focal_chair = focal_length_finder(KNOWN_DISTANCE, chair_WIDTH, rf.chair_width_in_rf )
                  #  distance = distance_finder(focal_chair, chair_WIDTH, width)
                   # distances[keyname] = distance

                if c == 24: # 5. Backpack
                    focal_backpack= focal_length_finder(KNOWN_DISTANCE, backpack_WIDTH, rf.backpack_width_in_rf )
                    distance = distance_finder(focal_backpack, backpack_WIDTH, width)
                    distances[keyname] = distance

                if c == 57: # 6. Couch
                    focal_couch= focal_length_finder(KNOWN_DISTANCE, couch_WIDTH, rf.couch_width_in_rf )
                    distance = distance_finder(focal_couch, couch_WIDTH, width)
                    distances[keyname] = distance

                if c == 63: # 7. Laptop
                    focal_laptop= focal_length_finder(KNOWN_DISTANCE, laptop_WIDTH, rf.laptop_width_in_rf )
                    distance = distance_finder(focal_laptop, laptop_WIDTH, width)
                    distances[keyname] = distance

                if c == 58: # 8. PottedPlant
                    focal_plant = focal_length_finder(KNOWN_DISTANCE, potted_plant_WIDTH, rf.potted_plant_width_in_rf )
                    distance = distance_finder(focal_plant, potted_plant_WIDTH, width)
                    distances[keyname] = distance

                if c == 62: # 9. TV
                    focal_TV = focal_length_finder(KNOWN_DISTANCE, tv_WIDTH, rf.tv_width_in_rf )
                    distance = distance_finder(focal_TV, tv_WIDTH, width)
                    distances[keyname] = distance

                if c == 60: # 10. Dining Table
                    focal_dining_table = focal_length_finder(KNOWN_DISTANCE, dining_table_WIDTH, rf.dining_table_in_rf )
                    distance = distance_finder(focal_dining_table, dining_table_WIDTH, width)
                    distances[keyname] = distance

                if c == 28: # 11. Suitcase
                    focal_suitcase = focal_length_finder(KNOWN_DISTANCE, suitcase_WIDTH, rf.suitcase_width_in_rf )
                    distance = distance_finder(focal_suitcase, suitcase_WIDTH, width)
                    distances[keyname] = distance

                if c == 2: # 1. Car
                    focal_car = focal_length_finder(KNOWN_DISTANCE, car_WIDTH, rf.car_width_in_rf )
                    distance = distance_finder(focal_car, car_WIDTH, width)
                    distances[keyname] = distance

                if c == 3: # 2. MotorCycle
                    focal_motorcycle = focal_length_finder(KNOWN_DISTANCE, motorcycle_WIDTH, rf.motorcycle_width_in_rf )
                    distance = distance_finder(focal_motorcycle, motorcycle_WIDTH, width)
                    distances[keyname] = distance

                if c == 1: # 3. Bicycle
                    focal_bicycle = focal_length_finder(KNOWN_DISTANCE, bicycle_WIDTH, rf.bicycle_width_in_rf )
                    distance = distance_finder(focal_bicycle, bicycle_WIDTH, width)
                    distances[keyname] = distance

                if c == 11: # 4. Stopsign
                    focal_stopsign = focal_length_finder(KNOWN_DISTANCE, stopsign_WIDTH, rf.stopsign_width_in_rf )
                    distance = distance_finder(focal_stopsign, stopsign_WIDTH, width)
                    distances[keyname] = distance

                if c == 9: # 5. Traffic Light
                    focal_trafficlight = focal_length_finder(KNOWN_DISTANCE, traffic_light_WIDth, rf.trafficlight_width_in_rf )
                    distance = distance_finder(focal_trafficlight, traffic_light_WIDth, width)
                    distances[keyname] = distance

                if c == 12: # 6. Parking Meter
                    focal_parkingmeter = focal_length_finder(KNOWN_DISTANCE, parking_meter_WIDTH, rf.parkingmeter_width_in_rf )
                    distance = distance_finder(focal_parkingmeter, parking_meter_WIDTH, width)
                    distances[keyname] = distance



            #find position of nearest detected item
            # use the center (x, y)-coordinates to derive the top and
			# and left corner of the bounding box
            x = int(x - (width/2))
            y = int(y - (height/2))
            if x <= W/3:
                W_pos = "left "
            elif x <= (W/3 * 2):
                W_pos = "center "
            else:
                W_pos = "right "

            #if y <= H/3:
            #    H_pos = "top "
            #elif y <= (H/3 * 2):
            #    H_pos = "mid "
            #else:
            #    H_pos = "bottom "

            position_key_name = f'{name}-{i}'
            positions[position_key_name] = f'{W_pos}'
            if distances:
                print(f'----------------Loop [{i}] OBJECT : {self.model.names[int(c)]} ---- {distance}')



        if distances:

            #get nearest object name distance from dictionary
            #min_k, min_v = min(distances.items(), key=lambda x: x[1])
            print(f'DICTIONARY :{distances}')

            min_k = min(distances, key=distances.get)
            min_v = distances.get(min_k)
            print(f'Key Name : {min_k} {min_v}')

            object_name = min_k.split("-")[0]

            print(f'------------NEAREST OBJECT name ------{object_name}')
            print(f'------------NEAREST OBJECT distance------ {min_v.item()}')


            self.new_text += f"And Nearest object is {object_name} at {positions.get(str(min_k))} in {min_v.item():.2f} meter"
            distances = {}

        if self.new_text != self.old_text:
            #Write to the file speech single line by line for realtime

            self.old_text = self.new_text
            speech_file = open('speech.txt','w')
            speech_file.write(self.new_text)
            speech_file.close()
            self.new_text = ''


        # write
        for d in reversed(det):

            c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())
            if c in selected_detected_class:

                if self.args.save_txt:  # Write to file
                    line = (c, *d.xywhn.view(-1)) + (conf, ) * self.args.save_conf + (() if id is None else (id, ))
                    with open(f'{self.txt_path}.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')
                if self.args.save or self.args.show:  # Add bbox to image
                    name = ('' if id is None else f'id:{id}') + self.model.names[c]
                    label = None if self.args.hide_labels else (name if self.args.hide_conf else f'{name} {conf:.2f}')
                    self.annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))

                if self.args.save_crop:
                    save_one_box(d.xyxy,
                                imc,
                                file=self.save_dir / 'crops' / self.model.names[c] / f'{self.data_path.stem}.jpg',
                                BGR=True)

        return log_string


def predict(cfg=DEFAULT_CFG, use_python=False):
    model = cfg.model or 'yolov8m.pt'
    source = cfg.source if cfg.source is not None else ROOT / 'assets' if (ROOT / 'assets').exists() \
        else 'https://ultralytics.com/images/bus.jpg'

    args = dict(model=model, source=source)
    if use_python:
        from ultralytics import YOLO
        YOLO(model)(**args)
    else:
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()


if __name__ == '__main__':
    predict()
