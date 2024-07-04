from tratamientoVideo import guardarVideo, leerVideo
from tracker import Tracker
from asignarEquipos import TeamAssigner
from asignarBalon import PlayerBallAssigner
import numpy as np
import os
import hashlib
from flask import Flask, request, jsonify, render_template, send_from_directory, Response
from werkzeug.utils import secure_filename
from threading import Thread, Lock
import cv2
import queue

app = Flask(__name__)

video_capture = cv2.VideoCapture(0)
frame_queue = queue.Queue()
processed_frames = []
capturing = False
frame_lock = Lock()

# Configuración
UPLOAD_FOLDER = 'inputs'
RESULTS_FOLDER = 'resultados'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Variables globales
checkbox_values = {}
uploaded_video_path = None
uploaded_video_name = None

# Crea el directorio de subida si no existe
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def capture_frames():
    global capturing
    while capturing:
        success, frame = video_capture.read()
        if success:
            frame_queue.put(frame)

def process_frames():
    global capturing
    tracker = Tracker('modelos/best.pt')
    detecciones = []
    while capturing or not frame_queue.empty():
        if not frame_queue.empty():
            frame = frame_queue.get()
            # Detectar
            detecciones += tracker.detectarRT(frame)
            with frame_lock:
                processed_frames.append(frame)
    tracks = tracker.obtenerTrayectoriasRT(detecciones)
    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(processed_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(processed_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Dibujar anotaciones
    output_frames = tracker.anotarRT(processed_frames, tracks)
    print('Guardando video...')
    guardarVideo(output_frames, 'output.avi')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def calculate_hash(file):
    hash_md5 = hashlib.md5()
    for chunk in iter(lambda: file.read(4096), b""):
        hash_md5.update(chunk)
    file.seek(0)  # Reset file pointer to the beginning
    return hash_md5.hexdigest()

def gen_frames():
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    global uploaded_video_path  # Declarar la variable como global para modificarla
    global uploaded_video_name  # Declarar la variable como global para modificarla

    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'})

    if file and allowed_file(file.filename):
        file_hash = calculate_hash(file)
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        filename = secure_filename(f"{file_hash[:8]}.{file_extension}")
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Actualizar la variable global con la ruta del video subido
        uploaded_video_path = file_path
        uploaded_video_name = file_hash[:8]

        return jsonify({'success': True, 'message': 'File uploaded successfully', 'filename': filename})

    return jsonify({'success': False, 'message': 'File type not allowed'})


@app.route('/inputs/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/start', methods=['POST'])
def start_action():
    # Se obtienen los valores de las opciones
    global checkbox_values
    checkbox_values = request.json
    print(checkbox_values)

    # Se lee al video a procesar
    frames = leerVideo(uploaded_video_path)

    # Inicializar Tracker
    tracker = Tracker('modelos/best.pt')
    # Buscar si existen las detecciones para el vídeo, si no se realizan
    tracks = tracker.obtenerTrayectorias(frames, read_from_detec=True, detec_path = f'detecciones/{uploaded_video_name}.pkl')

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Assign Ball Aquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = [1]
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)

    # Dibujar anotaciones
    output_frames = tracker.anotar(frames, tracks, team_ball_control, checkbox_values)

    # Se guarda el vídeo procesado
    guardarVideo(output_frames, f'resultados/{uploaded_video_name}.avi')

    return jsonify({'success': True, 'message': 'Valores de checkboxes recibidos', 'checkbox_values': checkbox_values})


@app.route('/results', methods=['POST'])
def get_processed_file():
    data = request.get_json()
    processed_filename = data['filename']
    processed_path = os.path.join(RESULTS_FOLDER, processed_filename)

    if os.path.exists(processed_path):
        return jsonify(success=True)
    else:
        return jsonify(success=False, message='Archivo procesado no encontrado.')


@app.route('/resultados/<filename>')
def uploaded_file2(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/start_capture')
def start_capture():
    global capturing, processed_frames
    capturing = True
    processed_frames = []
    capture_thread = Thread(target=capture_frames)
    process_thread = Thread(target=process_frames)
    capture_thread.start()
    process_thread.start()
    return "Capture started"

@app.route('/stop_capture')
def stop_capture():
    global capturing
    capturing = False
    return "Capture stopped"

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
