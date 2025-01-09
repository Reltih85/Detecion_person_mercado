from collections import deque
import csv
import cv2

class Logger:
    def __init__(self):
        self.logs = {
            'Zona Roja': deque(maxlen=100),
            'Zona Verde': deque(maxlen=100)
        }

    def add_log(self, zone, person_id, timestamp, interval):
        self.logs[zone].append((person_id, timestamp, interval))

    def display_logs(self, frame):
        for zone, color, y_offset in [('Zona Roja', (0, 0, 255), 30), ('Zona Verde', (0, 255, 0), 150)]:
            cv2.putText(frame, f"{zone}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            for i, (person_id, timestamp, interval) in enumerate(list(self.logs[zone])[-3:]):
                text = f"ID: {person_id}, Hora: {timestamp}, Intervalo: {interval}s"
                y_pos = y_offset + 25 * (i + 1)
                cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
