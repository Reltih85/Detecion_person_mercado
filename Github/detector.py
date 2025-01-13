import warnings
import torch
from tracker import Sort
from helpers import draw_detections, is_inside_zone
from ventana import Logger
from datetime import datetime
import pymysql
import cv2  # Asegúrate de tener OpenCV instalado

warnings.filterwarnings("ignore", category=FutureWarning)

class Detector:
    def __init__(self, model_name="yolov5n", confidence_threshold=0.10):
        self.model = torch.hub.load("ultralytics/yolov5", model=model_name, pretrained=True)
        self.confidence_threshold = confidence_threshold
        self.tracker = Sort()
        self.logger = Logger()
        self.arrival_times = {}
        self.departure_times = {}
        self.service_start_times = {}
        self.service_end_times = {}

        # Conexión e inicialización de la base de datos
        self.connection = self.connect_to_db()
        self.initialize_database()

    def connect_to_db(self):
        try:
            connection = pymysql.connect(
                host="localhost",
                user="root",  # Cambia esto si tienes un usuario diferente
                password="",  # Cambia esto si tienes una contraseña configurada
                database="tracking",  # Nombre de la base de datos
                charset="utf8mb4",
                cursorclass=pymysql.cursors.DictCursor
            )
            print("Conexión exitosa al servidor de base de datos.")
            return connection
        except pymysql.MySQLError as e:
            print(f"Error al conectar al servidor de base de datos: {e}")
            return None

    def initialize_database(self):
        """Crea las tablas necesarias en la base de datos si no existen."""
        if not self.connection:
            print("No se puede inicializar la base de datos. Conexión no disponible.")
            return

        try:
            with self.connection.cursor() as cursor:
                # Crear tablas si no existen
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS zona_roja (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    id_persona INT NOT NULL,
                    tentrada TIME NOT NULL,
                    tsalida TIME NOT NULL
                )
                """)

                cursor.execute("""
                CREATE TABLE IF NOT EXISTS zona_verde (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    id_persona INT NOT NULL,
                    tentrada TIME NOT NULL,
                    sdurante VARCHAR(20) NOT NULL,
                    tsalida TIME NOT NULL
                )
                """)

                self.connection.commit()
                print("Base de datos y tablas inicializadas correctamente.")
        except pymysql.MySQLError as e:
            print(f"Error al inicializar la base de datos: {e}")

    def save_to_db(self, table, data):
        """Guarda datos en la tabla especificada."""
        if not self.connection:
            print("No se puede guardar en la base de datos. Conexión no disponible.")
            return

        try:
            with self.connection.cursor() as cursor:
                if table == "zona_roja":
                    query = "INSERT INTO zona_roja (id_persona, tentrada, tsalida) VALUES (%s, %s, %s)"
                    cursor.execute(query, data)
                elif table == "zona_verde":
                    query = "INSERT INTO zona_verde (id_persona, tentrada, sdurante, tsalida) VALUES (%s, %s, %s, %s)"
                    cursor.execute(query, data)

                self.connection.commit()
                print(f"Datos guardados en la tabla {table} correctamente.")
        except pymysql.MySQLError as e:
            print(f"Error al guardar en la base de datos: {e}")

    def get_bboxes(self, preds, name_filter):
        """Obtiene las coordenadas de los objetos detectados que cumplen el filtro."""
        df = preds.pandas().xyxy[0]
        return df[(df["confidence"] >= self.confidence_threshold) & (df["name"] == name_filter)][['xmin', 'ymin', 'xmax', 'ymax']].values

    def process_frame(self, frame, zones):
        """Procesa cada cuadro detectando personas y manejando las zonas."""
        preds = self.model(frame)
        frame_data = self.tracker.update(self.get_bboxes(preds, "person"))

        for obj in frame_data:
            x1, y1, x2, y2, tid = obj
            xc, yc = draw_detections(frame, obj, (0, 0, 255), (255, 255, 255), is_person=True)

            # Zona Roja (Llegada y Salida)
            if is_inside_zone((xc, yc), zones['arrival_departure']):
                if tid not in self.arrival_times:
                    self.arrival_times[tid] = datetime.now().time()
            else:
                if tid in self.arrival_times and tid not in self.departure_times:
                    # Registrar la hora de salida
                    self.departure_times[tid] = datetime.now().time()

                    # Calcular la duración (intervalo)
                    entrada = datetime.combine(datetime.today(), self.arrival_times[tid])
                    salida = datetime.combine(datetime.today(), self.departure_times[tid])
                    duration = salida - entrada

                    # Convertir la duración a un formato legible (horas: minutos: segundos)
                    interval = str(duration).split('.')[0]  # Esto elimina los microsegundos y te deja solo horas, minutos, segundos

                    # Guardar en la base de datos con la duración calculada
                    self.save_to_db("zona_roja", (tid, self.arrival_times.pop(tid), self.departure_times[tid]))

                    # Agregar el log correspondiente con el intervalo
                    self.logger.add_log(
                        zone='Zona Roja',
                        person_id=tid,
                        timestamp=self.departure_times[tid].strftime("%H:%M:%S"),  # Aquí se asegura que solo horas, minutos y segundos
                        interval=interval  # Intervalo sin microsegundos
                    )


            # Zona Verde (Servicio)
            if is_inside_zone((xc, yc), zones['service']):
                if tid not in self.service_start_times:
                    self.service_start_times[tid] = datetime.now().time()
            else:
                if tid in self.service_start_times and tid not in self.service_end_times:
                    entrada = self.service_start_times[tid]
                    self.service_end_times[tid] = datetime.now().time()

                    # Calcular la duración (intervalo)
                    entrada_datetime = datetime.combine(datetime.today(), entrada)
                    salida_datetime = datetime.combine(datetime.today(), self.service_end_times[tid])
                    duration = salida_datetime - entrada_datetime

                    # Convertir la duración a un formato legible (horas: minutos: segundos)
                    service_duration = str(duration).split('.')[0]  # Eliminar los microsegundos

                    # Guardar en la base de datos con la duración calculada
                    self.save_to_db("zona_verde", (tid, entrada, service_duration, self.service_end_times[tid]))

                    # Agregar el log correspondiente con el intervalo
                    self.logger.add_log(
                        zone='Zona Verde',
                        person_id=tid,
                        timestamp=self.service_end_times[tid].strftime("%H:%M:%S"),  # Aquí se asegura que solo horas, minutos y segundos
                        interval=service_duration  # Intervalo sin microsegundos
                    )

                    # Eliminar la hora de entrada después de usarla
                    self.service_start_times.pop(tid)



        self.logger.display_logs(frame)
        return frame
