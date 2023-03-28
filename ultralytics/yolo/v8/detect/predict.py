# Ultralytics YOLO 🚀, GPL-3.0 license

import torch

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
# from gtts import gTTS
import ReferenceImageVal as rf
new_text =''
old_text = ''

#DISTANCE CONTASTANT
KNOWN_DISTANCE = 1 # meter
cell_phone_WIDTH = 0.08 #meter
person_WIDTH = 0.40 #meter
backpack_WIDTH = 0.35
handbag_WIDTH = 0.26
bottle_WIDTH = 0.06
chair_WIDTH = 0.5
dining_table_WIDTH = 0.96
laptop_WIDTH = 0.35
mouse_WIDTH = 0.08
keyboard_WIDTH = 0.42
book_WIDTH = 0.12

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
            return f'{log_string}(no detections), '


        (H,W) = im.shape[:2]
        distance = 0

        for c in det.cls.unique():
            n = (det.cls == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
            self.new_text += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

        distances = {}
        for i, box in enumerate(reversed(det)):

            print(f'-------CONFIDENCE SCORE---{box} ---------')
            x, y, width, height = det.xywh[i]
            c = int(det.cls[i])
            if c == 0 :
                focal_person = focal_length_finder(KNOWN_DISTANCE, person_WIDTH, rf.person_width_in_rf)
                distance = distance_finder(focal_person, person_WIDTH, width)
                distances[self.model.names[c]] = distance

            if c == 67:
                focal_mobile = focal_length_finder(KNOWN_DISTANCE, cell_phone_WIDTH, rf.mobile_width_in_rf )
                distance = distance_finder(focal_mobile, cell_phone_WIDTH, width)
                distances[self.model.names[c]] = distance

            if c == 26:
                focal_handbag= focal_length_finder(KNOWN_DISTANCE, handbag_WIDTH, rf.handbag_width_in_rf )
                distance = distance_finder(focal_handbag, handbag_WIDTH, width)
                distances[self.model.names[c]] = distance

            if c == 64:
                focal_mouse= focal_length_finder(KNOWN_DISTANCE, mouse_WIDTH, rf.mouse_width_in_rf )
                distance = distance_finder(focal_mouse, mouse_WIDTH, width)
                distances[self.model.names[c]] = distance

            if c == 24:
                focal_backpack= focal_length_finder(KNOWN_DISTANCE, backpack_WIDTH, rf.backpack_width_in_rf )
                distance = distance_finder(focal_backpack, backpack_WIDTH, width)
                distances[self.model.names[c]] = distance

            if c == 39:
                focal_bottle= focal_length_finder(KNOWN_DISTANCE, bottle_WIDTH, rf.bottle_width_in_rf )
                distance = distance_finder(focal_bottle, bottle_WIDTH, width)
                distances[self.model.names[c]] = distance

            if c == 63:
               focal_laptop= focal_length_finder(KNOWN_DISTANCE, laptop_WIDTH, rf.laptop_width_in_rf )
               distance = distance_finder(focal_laptop, laptop_WIDTH, width)
               distances[self.model.names[c]] = distance

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

            if y <= H/3:
                H_pos = "top "
            elif y <= (H/3 * 2):
                H_pos = "mid "
            else:
                H_pos = "bottom "

            #self.new_text += f"And Nearest object is {self.model.names[c]} at {H_pos}  {W_pos} in {distance:.2f} meter"
            #break

        if distances:
            for i in distances:
                print(f'---------Distance : {i}')
            print('------------NEAREST OBJECT ------'+min(distances, key=distances.get))
            print('------------NEAREST OBJECT distance------'+str(min(distances.values())))
            nearest_object_name = min(distances, key=distances.get)
            nearest_object_distance = min(distances.values())
            #at {H_pos}  {W_pos}
            self.new_text += f"And Nearest object is {nearest_object_name} in {nearest_object_distance:.2f} meter"

        if self.new_text != self.old_text:
            #Write to the file
            self.old_text = self.new_text
            file = open('speech.txt','w')
            a = file.write(self.new_text)
            file.close()

            # #Generate mp3 file
            # gtts = gTTS(self.new_text, lang='en')
            # gtts.save('myaudio.mp3')

            #reset the text
            self.new_text = ''

        # write
        for d in reversed(det):

            c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())
            print(f'---------Confidence interval {conf}')

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
    model = cfg.model or 'yolov8n.pt'
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
