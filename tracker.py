import numpy as np

class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.bbox = bbox
        self.hits = 0
        self.age = 0
        self.time_since_update = 0

    def update(self, bbox):
        self.bbox = bbox
        self.time_since_update = 0
        self.hits += 1

    def predict(self):
        self.time_since_update += 1
        self.age += 1
        return self.bbox

class Sort:
    def __init__(self, max_age=30, min_hits=2):
        self.trackers = []
        self.max_age = max_age
        self.min_hits = min_hits

    def update(self, dets=np.empty((0, 5))):
        for trk in self.trackers:
            trk.predict()

        used_tracks = []
        for det in dets:
            assigned = False
            for t in self.trackers:
                if t.time_since_update > self.max_age:
                    continue
                if self._iou(t.bbox, det[:4]) > 0.4 and t not in used_tracks:
                    t.update(det[:4])
                    assigned = True
                    used_tracks.append(t)
                    break
            if not assigned:
                self.trackers.append(KalmanBoxTracker(det[:4]))

        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        return np.array([[*t.bbox, t.id] for t in self.trackers if t.hits >= self.min_hits])

    @staticmethod
    def _iou(bb_test, bb_gt):
        xx1, yy1 = max(bb_test[0], bb_gt[0]), max(bb_test[1], bb_gt[1])
        xx2, yy2 = min(bb_test[2], bb_gt[2]), min(bb_test[3], bb_gt[3])
        w, h = max(0., xx2 - xx1 + 1), max(0., yy2 - yy1 + 1)
        wh = w * h
        return wh / ((bb_test[2] - bb_test[0] + 1) * (bb_test[3] - bb_test[1] + 1) +
                     (bb_gt[2] - bb_gt[0] + 1) * (bb_gt[3] - bb_gt[1] + 1) - wh)
