import cv2
from zones import ZONES
from helpers import draw_zones
from detector import Detector

def main():
    detector = Detector()  # Inicializa el detector con la lógica de detección.mys
    cap = cv2.VideoCapture("data/persons.mp4")  # Carga el video de entrada.

    try:
        while cap.isOpened():
            status, frame = cap.read()
            if not status:
                break

            # Procesa el frame para detectar personas en las zonas definidas.
            frame = detector.process_frame(frame, ZONES)
            # Dibuja las zonas de interés sobre el frame procesado.
            draw_zones(frame, ZONES, [(0, 0, 255), (0, 255, 0)])

            cv2.imshow("Detector de Personas", frame)  # Muestra el resultado en tiempo real.
            if cv2.waitKey(10) & 0xFF == ord('q'):  # Permite salir con la tecla 'q'.
                break

    except Exception as e:
        print(f"Error durante la ejecución: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Programa terminado.")  # Libera recursos y finaliza.

if __name__ == "__main__":
    main()
