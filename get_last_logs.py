import csv
import cv2

class Logger:
    def __init__(self):
        self.logs = []

    def add_log(self, zone, person_id, timestamp, duration):
        self.logs.append({"zone": zone, "person_id": person_id, "timestamp": timestamp, "duration": duration})

    def display_logs(self, frame):
        for i, log in enumerate(self.logs[-3:]):  # Mostrar los últimos 3 registros
            y_offset = 30 * (i + 1)
            log_text = f"[{log['zone']}] ID: {log['person_id']}, Time: {log['timestamp']}, Duration: {log['duration']}s"
            cv2.putText(frame, log_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def get_last_logs(self, count=3):
        return self.logs[-count:]

    def save_to_csv(self, filename):
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['zone', 'person_id', 'timestamp', 'duration']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.logs)
