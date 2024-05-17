# import numpy as np
# from filterpy.kalman import KalmanFilter
# from scipy.optimize import linear_sum_assignment
#
#
# def iou(bb_test, bb_gt):
#     xx1 = np.maximum(bb_test[0], bb_gt[0])
#     yy1 = np.maximum(bb_test[1], bb_gt[1])
#     xx2 = np.minimum(bb_test[2], bb_gt[2])
#     yy2 = np.minimum(bb_test[3], bb_gt[3])
#     w = np.maximum(0., xx2 - xx1)
#     h = np.maximum(0., yy2 - yy1)
#     wh = w * h
#     o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) +
#               (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
#     return o
#
#
# class KalmanBoxTracker:
#     def __init__(self, bbox, obj_id, tracker_id):
#         self.kf = KalmanFilter(dim_x=7, dim_z=4)
#         self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
#                               [0, 1, 0, 0, 0, 1, 0],
#                               [0, 0, 1, 0, 0, 0, 1],
#                               [0, 0, 0, 1, 0, 0, 0],
#                               [0, 0, 0, 0, 1, 0, 0],
#                               [0, 0, 0, 0, 0, 1, 0],
#                               [0, 0, 0, 0, 0, 0, 1]])
#         self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
#                               [0, 1, 0, 0, 0, 0, 0],
#                               [0, 0, 0, 1, 0, 0, 0],
#                               [0, 0, 0, 0, 0, 1, 0]])
#         self.kf.R[2:, 2:] *= 10.
#         self.kf.P[4:, 4:] *= 1000.
#         self.kf.P *= 10.
#         self.kf.Q[-1, -1] *= 0.01
#         self.kf.Q[4:, 4:] *= 0.01
#         self.kf.x[:4] = self.convert_bbox_to_z(bbox)
#         self.time_since_update = 0
#         self.obj_id = obj_id
#         self.tracker_id = tracker_id
#         self.history = []
#         self.hits = 0
#         self.hit_streak = 0
#         self.age = 0
#
#     def update(self, bbox):
#         self.time_since_update = 0
#         self.history = []
#         self.hits += 1
#         self.hit_streak += 1
#         self.kf.update(self.convert_bbox_to_z(bbox))
#
#     def predict(self):
#         if (self.kf.x[6] + self.kf.x[2]) <= 0:
#             self.kf.x[6] *= 0.0
#         self.kf.predict()
#         self.age += 1
#         if self.time_since_update > 0:
#             self.hit_streak = 0
#         self.time_since_update += 1
#         self.history.append(self.convert_x_to_bbox(self.kf.x))
#         return self.history[-1]
#
#     def get_state(self):
#         return self.convert_x_to_bbox(self.kf.x)
#
#     @staticmethod
#     def convert_bbox_to_z(bbox):
#         w = bbox[2] - bbox[0]
#         h = bbox[3] - bbox[1]
#         x = bbox[0] + w / 2.
#         y = bbox[1] + h / 2.
#         s = w * h
#         r = w / float(h)
#         return np.array([x, y, s, r]).reshape((4, 1))
#
#     @staticmethod
#     def convert_x_to_bbox(x, score=None):
#         w = np.sqrt(x[2] * x[3])
#         h = x[2] / w
#         if score is None:
#             return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
#         else:
#             return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))
#
#
# class Sort:
#     def __init__(self, max_age=10, min_hits=3, iou_threshold=0.3):
#         self.max_age = max_age  # 增加 max_age 以确保卡尔曼滤波器有足够的时间收敛
#         self.min_hits = min_hits
#         self.iou_threshold = iou_threshold
#         self.trackers = []
#         self.frame_count = 0
#         self.next_tracker_id = 0  # 全局唯一的跟踪器ID
#
#     def update(self, dets=np.empty((0, 5)), ids=np.empty((0,))):
#         self.frame_count += 1
#         trks = np.zeros((len(self.trackers), 5))
#         to_del = []
#         ret = []
#         for t, trk in enumerate(trks):
#             pos = self.trackers[t].predict()[0]
#             trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
#             if np.any(np.isnan(pos)):
#                 to_del.append(t)
#         trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
#         for t in reversed(to_del):
#             self.trackers.pop(t)
#
#         # 打印检测结果和跟踪器预测位置
#         print("Detections: ", dets)
#         print("Trackers: ", trks)
#
#         matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)
#         print(f"Matched: {matched}, Unmatched Dets: {unmatched_dets}, Unmatched Trks: {unmatched_trks}")
#
#         for m in matched:
#             print(f"Updating tracker {m[1]} with detection {m[0]}")
#             self.trackers[m[1]].update(dets[m[0], :])
#         for i in unmatched_dets:
#             trk = KalmanBoxTracker(dets[i, :5], ids[i], self.next_tracker_id)
#             self.next_tracker_id += 1
#             print(f"Creating new tracker {trk.tracker_id} for detection {i}")
#             self.trackers.append(trk)
#         i = len(self.trackers)
#         for trk in reversed(self.trackers):
#             d = trk.get_state()[0]
#             if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
#                 ret.append(np.concatenate((d, [trk.tracker_id])).reshape(1, -1))
#             i -= 1
#             if trk.time_since_update > self.max_age:
#                 print(f"Removing tracker {trk.tracker_id} due to age")
#                 self.trackers.pop(i)
#         if len(ret) > 0:
#             return np.concatenate(ret)
#         return np.empty((0, 5))
#
#
# def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
#     if len(trackers) == 0:
#         return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
#     iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32



import numpy as np
from database import get_max_tracker_id

class SimpleTracker:
    def __init__(self, bbox, yolo_id, tracker_id):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.yolo_id = yolo_id
        self.tracker_id = tracker_id
        self.time_since_update = 0

    def update(self, bbox):
        self.bbox = bbox
        self.time_since_update = 0

    def predict(self):
        self.time_since_update += 1

    def get_state(self):
        return self.bbox, self.tracker_id

class Sort:
    def __init__(self, max_age=10, iou_threshold=0.3):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.next_tracker_id = get_max_tracker_id()

    def _iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def update(self, dets=np.empty((0, 5)), ids=np.empty((0,))):
        if len(dets) != len(ids):
            raise ValueError(f"Size mismatch: dets size = {len(dets)}, ids size = {len(ids)}")

        self.frame_count += 1
        ret = []

        # 预测现有跟踪器的位置
        for tracker in self.trackers:
            tracker.predict()

        # 用检测结果更新跟踪器
        for i, det in enumerate(dets):
            yolo_id = ids[i]
            matched = False
            for tracker in self.trackers:
                if tracker.yolo_id == yolo_id and self._iou(det[:4], tracker.bbox) >= self.iou_threshold:
                    tracker.update(det[:4])
                    matched = True
                    break
            if not matched:
                # 创建新的跟踪器
                tracker = SimpleTracker(det[:4], yolo_id, self.next_tracker_id)
                self.next_tracker_id += 1
                self.trackers.append(tracker)

        # 移除长时间未更新的跟踪器
        self.trackers = [tracker for tracker in self.trackers if tracker.time_since_update <= self.max_age]

        # 收集跟踪结果
        for tracker in self.trackers:
            bbox, tracker_id = tracker.get_state()
            ret.append(np.concatenate((bbox, [tracker_id], [tracker.yolo_id])).reshape(1, -1))

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 6))
