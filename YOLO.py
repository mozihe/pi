import cv2
import numpy as np
import onnxruntime as ort
from random import randint

class YOLO:
    def __init__(self, model_path, labels, model_h, model_w, strides, anchors, thred_nms=0.4, thred_cond=0.1):
        self.model_h = model_h
        self.model_w = model_w
        self.strides = strides
        self.anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(len(strides), -1, 2)
        self.labels = labels
        self.thred_nms = thred_nms
        self.thred_cond = thred_cond

        so = ort.SessionOptions()
        self.net = ort.InferenceSession(model_path, so)

    @staticmethod
    def plot_one_box(x, img, color=None, label=None, line_thickness=None):
        tl = (line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1)
        color = color or [randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    def _make_grid(self, nx, ny):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

    def preprocess(self, img):
        img_resized = cv2.resize(img, (self.model_w, self.model_h), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        blob = np.expand_dims(np.transpose(img_normalized, (2, 0, 1)), axis=0)
        return blob

    def postprocess(self, outputs, img_h, img_w):
        conf = outputs[:, 4]
        c_x = outputs[:, 0] / self.model_w * img_w
        c_y = outputs[:, 1] / self.model_h * img_h
        w = outputs[:, 2] / self.model_w * img_w
        h = outputs[:, 3] / self.model_h * img_h
        p_cls = outputs[:, 5:]
        cls_id = np.argmax(p_cls, axis=1)
        p_x1 = np.expand_dims(c_x - w / 2, -1)
        p_y1 = np.expand_dims(c_y - h / 2, -1)
        p_x2 = np.expand_dims(c_x + w / 2, -1)
        p_y2 = np.expand_dims(c_y + h / 2, -1)
        areas = np.concatenate((p_x1, p_y1, p_x2, p_y2), axis=-1)

        areas = areas.tolist()
        indices = cv2.dnn.NMSBoxes(areas, conf.tolist(), self.thred_cond, self.thred_nms)
        if len(indices) > 0:
            return np.array(areas)[indices], np.array(conf)[indices], cls_id[indices]
        else:
            return [], [], []

    def cal_outputs(self, outs):
        row_ind = 0
        grid = [np.zeros(1)] * len(self.strides)
        for i in range(len(self.strides)):
            h, w = int(self.model_w / self.strides[i]), int(self.model_h / self.strides[i])
            length = int(len(self.anchor_grid[i]) * h * w)
            if grid[i].shape[2:4] != (h, w):
                grid[i] = self._make_grid(w, h)
            outs[row_ind:row_ind + length, 0:2] = (outs[row_ind:row_ind + length, 0:2] * 2. - 0.5 + np.tile(
                grid[i], (len(self.anchor_grid[i]), 1))) * int(self.strides[i])
            outs[row_ind:row_ind + length, 2:4] = (outs[row_ind:row_ind + length, 2:4] * 2) ** 2 * np.repeat(
                self.anchor_grid[i], h * w, axis=0)
            row_ind += length
        return outs

    def infer_img(self, img):
        blob = self.preprocess(img)
        outs = self.net.run(None, {self.net.get_inputs()[0].name: blob})[0].squeeze(axis=0)
        outs = self.cal_outputs(outs)
        img_h, img_w, _ = img.shape
        return self.postprocess(outs, img_h, img_w)

    def draw_detections(self, img, boxes, confs, ids):
        for box, score, id in zip(boxes, confs, ids):
            label = '%s:%.2f' % (self.labels[id], score)
            self.plot_one_box(box.astype(np.int16), img, color=(255, 0, 0), label=label, line_thickness=None)