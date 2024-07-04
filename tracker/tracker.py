from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
from tratamientoVideo import centroBbox, anchoBbox
import numpy as np
import pandas as pd


class Tracker:
    def __init__(self, path_modelo):
        self.model = YOLO(path_modelo)
        self.tracker = sv.ByteTrack()

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Dibujar un rectángulo con transparencia para poner la posesión
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        # Calcular el número de frames que cada equipo tiene el balón
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]
        team_1 = team_1_num_frames / (team_1_num_frames + team_2_num_frames)
        team_2 = team_2_num_frames / (team_1_num_frames + team_2_num_frames)

        cv2.putText(frame, f"Posesion equipo 1: {team_1 * 100:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 0), 3)
        cv2.putText(frame, f"Posesion equipo 2: {team_2 * 100:.2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 0), 3)

        return frame

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    # Función para realizar las detecciones pasándole la serie de frames
    def detectar(self, frames):
        lote = 20
        detecciones = []
        # El bucle se recorre por lotes para optimizar la memoria (de 20 en 20 frames) aunque dentro se procesan todas
        for i in range(0, len(frames), lote):
            lote_detecciones = self.model.predict(frames[i:i + lote], conf=0.1)
            detecciones += lote_detecciones

        return detecciones

    # Función para realizar las detecciones en tiempo real
    def detectarRT(self, frame):
        frame = self.model.predict(frame, conf=0.1)
        return frame

    def obtenerTrayectorias(self, frames, read_from_detec=False, detec_path=None):

        if read_from_detec and detec_path is not None and os.path.exists(detec_path):
            with open(detec_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        # Se realizan las detecciones
        detecciones = self.detectar(frames)

        # Se crea un diccionario con las trayectorias de jugadores, árbitros y balón
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for num_frame, deteccion in enumerate(detecciones):
            # Obtenemos el nombre de las clases
            names = deteccion.names
            # Nombre de las clases inverso, asociado al id
            names_inv = {v: k for k, v in names.items()}

            # Se convierten las detecciones al formato de Supervision
            deteccion_sv = sv.Detections.from_ultralytics(deteccion)

            # Transformar los porteros para tratarlos como jugadores
            for ind_detec, id_clase in enumerate(deteccion_sv.class_id):
                # Si el nombre de la clase por su id coincide con portero, se le asigna el id de jugador
                if names[id_clase] == "goalkeeper":
                    deteccion_sv.class_id[ind_detec] = names_inv["player"]

            # Seguir objetos
            deteccion_seguir = self.tracker.update_with_detections(deteccion_sv)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            # Se guardan las coordenadas de la bounding box, el id de la clase y del track
            for detecciones_frame in deteccion_seguir:
                bbox = detecciones_frame[0].tolist()
                id_clase = detecciones_frame[3]
                id_track = detecciones_frame[4]

                # Se hace con jugadores y árbitros ya que no se va a hacer seguimiento del balón (solo hay uno)
                if id_clase == names_inv['player']:
                    tracks["players"][num_frame][id_track] = {"bbox": bbox}

                if id_clase == names_inv['referee']:
                    tracks["referees"][num_frame][id_track] = {"bbox": bbox}

            # Igual con el balón (pero obviando los track_id)
            for detecciones_frame in deteccion_sv:
                bbox = detecciones_frame[0].tolist()
                id_clase = detecciones_frame[3]

                if id_clase == names_inv['ball']:
                    tracks["ball"][num_frame][1] = {"bbox": bbox}

        if detec_path is not None:
            with open(detec_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def obtenerTrayectoriasRT(self, detecciones):

        # Se crea un diccionario con las trayectorias de jugadores, árbitros y balón
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for num_frame, deteccion in enumerate(detecciones):
            # Obtenemos el nombre de las clases
            names = deteccion.names
            # Nombre de las clases inverso, asociado al id
            names_inv = {v: k for k, v in names.items()}

            # Se convierten las detecciones al formato de Supervision
            deteccion_sv = sv.Detections.from_ultralytics(deteccion)

            # Transformar los porteros para tratarlos como jugadores
            for ind_detec, id_clase in enumerate(deteccion_sv.class_id):
                # Si el nombre de la clase por su id coincide con portero, se le asigna el id de jugador
                if names[id_clase] == "goalkeeper":
                    deteccion_sv.class_id[ind_detec] = names_inv["player"]

            # Seguir objetos
            deteccion_seguir = self.tracker.update_with_detections(deteccion_sv)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            # Se guardan las coordenadas de la bounding box, el id de la clase y del track
            for detecciones_frame in deteccion_seguir:
                bbox = detecciones_frame[0].tolist()
                id_clase = detecciones_frame[3]
                id_track = detecciones_frame[4]

                # Se hace con jugadores y árbitros ya que no se va a hacer seguimiento del balón (solo hay uno)
                if id_clase == names_inv['player']:
                    tracks["players"][num_frame][id_track] = {"bbox": bbox}

                if id_clase == names_inv['referee']:
                    tracks["referees"][num_frame][id_track] = {"bbox": bbox}

            # Igual con el balón (pero obviando los track_id)
            for detecciones_frame in deteccion_sv:
                bbox = detecciones_frame[0].tolist()
                id_clase = detecciones_frame[3]

                if id_clase == names_inv['ball']:
                    tracks["ball"][num_frame][1] = {"bbox": bbox}

        return tracks

    # Función para señalar los jugadores con elipses
    def elipse(self, frame, bbox, color, id_track=None):
        y2 = int(bbox[3])
        centro_x, _ = centroBbox(bbox)
        width = anchoBbox(bbox)

        cv2.ellipse(
            frame,
            center=(centro_x, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = centro_x - rectangle_width // 2
        x2_rect = centro_x + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if id_track is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)

            x1_text = x1_rect + 12
            if id_track > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{id_track}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame

    def triangulo(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = centroBbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def anotar(self, frames_video, tracks, team_ball_control, checkbox_values):
        output_frames = []
        for num_frame, frame in enumerate(frames_video):
            frame = frame.copy()

            player_dict = tracks["players"][num_frame]
            ball_dict = tracks["ball"][num_frame]
            referee_dict = tracks["referees"][num_frame]

            # Anotar jugadores
            if checkbox_values['opcion1']:
                for id_track, player in player_dict.items():
                    color = player.get("team_color", (0, 0, 255))
                    frame = self.elipse(frame, player["bbox"], color, id_track)

                    if player.get('has_ball', False):
                        frame = self.triangulo(frame, player["bbox"], (0, 0, 255))

            # Draw Referee
            if checkbox_values['opcion3']:
                for _, referee in referee_dict.items():
                    frame = self.elipse(frame, referee["bbox"], (0, 255, 255))

            #    # Draw ball
            if checkbox_values['opcion2']:
                for track_id, ball in ball_dict.items():
                    frame = self.triangulo(frame, ball["bbox"], (0, 255, 0))

            #
            #    # Draw Team Ball Control
            if checkbox_values['opcion4']:
                frame = self.draw_team_ball_control(frame, num_frame, team_ball_control)

            output_frames.append(frame)

        return output_frames

    def anotarRT(self, frames_video, tracks):
        output_frames = []
        for num_frame, frame in enumerate(frames_video):
            frame = frame.copy()

            player_dict = tracks["players"][num_frame]
            ball_dict = tracks["ball"][num_frame]
            referee_dict = tracks["referees"][num_frame]

            # Anotar jugadores
            for id_track, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.elipse(frame, player["bbox"], color, id_track)

                if player.get('has_ball', False):
                    frame = self.triangulo(frame, player["bbox"], (0, 0, 255))
            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.elipse(frame, referee["bbox"], (0, 255, 255))

            #    # Draw ball
            for track_id, ball in ball_dict.items():
                frame = self.triangulo(frame, ball["bbox"], (0, 255, 0))

            output_frames.append(frame)

        return output_frames
