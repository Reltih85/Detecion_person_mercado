import cv2
from zones import ZONES
from helpers import draw_zones
from detector import Detector

def main():
    detector = Detector()
    cap = cv2.VideoCapture("data/persons.mp4")

    try:
        while cap.isOpened():
            status, frame = cap.read()
            if not status:
                break

            frame = detector.process_frame(frame, ZONES)
            draw_zones(frame, ZONES, [(0, 0, 255), (0, 255, 0)])

            cv2.imshow("Detector de Personas", frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error durante la ejecución: {e}")

    finally:
        # Liberar recursos
        cap.release()
        cv2.destroyAllWindows()
        print("Programa terminado.")

if __name__ == "__main__":
    main()
