# Este archivo contiene funciones genéricas para procesar los vídeos
# Estas funciones (o algunas muy parecidas) se pueden encontrar en muchos sitios,
# incluida la página oficial de openCV

import cv2

def leerVideo (path_video):
    # Se crea un objeto videoCapture para poder leer el vídeo
    cap = cv2.VideoCapture(path_video)
    # Se crea una lista vacía para ir almacenando los frames obtenidos
    frames = []
    # Un bulce para interar sobre cada frame del vídeo
    while True:
        # Read devuelve el propio frame y un valor 'ret' que indica si es el último frame o no
        ret, frame = cap.read()
        if not ret:
            # Si ret = False, se sale del bucle, el vídeo ha terminado
            break
        resized_frame = cv2.resize(frame, (1920, 1080))
        frames.append(resized_frame)
    return frames

def guardarVideo(frames_procesadas, path_video_procesado):
    # Se define el formato de salida
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # Se define como se van a guardar los frames: path, formato, frames/segundo, dimensiones
    out = cv2.VideoWriter(path_video_procesado, fourcc, 24, (frames_procesadas[0].shape[1], frames_procesadas[0].shape[0]))
    # Se itera por todas las frames procesadas y se usa write para ir componiendo el video
    for frame in frames_procesadas:
        out.write(frame)
    out.release()

# Función para obtener el centro de una bounding box
def centroBbox(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int((y1+y2)/2)

def anchoBbox(bbox):
    return bbox[2]-bbox[0]

def measure_distance(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def measure_xy_distance(p1,p2):
    return p1[0]-p2[0],p1[1]-p2[1]

def get_foot_position(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int(y2)