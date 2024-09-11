from time import sleep

import cv2
import time

import numpy as np
from flask import Flask, render_template, Response, request, jsonify, redirect, url_for, session
import psutil
from YOLO import YOLO
import logging
import threading
import os
import json
import sqlite3
from queue import Queue
from sort import Sort
from database import insert_new_tracking_record, update_tracking_record, init_db, check_if_track_exists
from datetime import datetime

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

app = Flask(__name__)
app.secret_key = 'woaishumeipai'  # 用于会话管理的密钥

ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'password'

init_db()

model_folder = "models"
label_folder = "labels"
model_file_path = os.path.join(model_folder, "best_lyl.onnx")
label_file_path = os.path.join(label_folder, "lyl.json")
yolo = None
cap = None
frame_queue = Queue(maxsize=1)
tracker = Sort()

time.sleep(1)


def initialize_yolo_and_camera():
    global yolo, cap
    with open(label_file_path, 'r') as f:
        labels = json.load(f)
        labels = {int(k): v for k, v in labels.items()}  # 将字符串键转换为整数

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Cannot open camera")
        exit(-1)

    model_h = 320
    model_w = 320
    strides = [8., 16., 32.]
    anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]

    yolo = YOLO(model_file_path, labels, model_h, model_w, strides, anchors)


def yolo_inference():
    global tracker, cap, frame_queue
    while True:
        success, img0 = cap.read()

        if not success:
            logging.error("Failed to read frame from camera")
            sleep(0.01)
            continue

        t1 = time.time()
        det_boxes, scores, ids = yolo.infer_img(img0)
        if len(det_boxes) > 0:
            detections = np.hstack((det_boxes, scores.reshape(-1, 1)))
            tracking_results = tracker.update(detections, ids)
        else:
            tracking_results = tracker.update()

        # 绘制跟踪结果
        for res in tracking_results:
            x1, y1, x2, y2, tracker_id, yolo_id = res
            cv2.rectangle(img0, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(img0, f'ID: {int(tracker_id)}, Label: {yolo.labels[yolo_id]}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if not check_if_track_exists(tracker_id):
                insert_new_tracking_record(tracker_id, yolo.labels[yolo_id], timestamp)
            else:
                update_tracking_record(tracker_id, timestamp)

        t2 = time.time()
        str_FPS = "FPS: %.2f" % (1. / (t2 - t1))
        cv2.putText(img0, str_FPS, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

        if frame_queue.full():
            frame_queue.get()
        frame_queue.put(cv2.imencode('.jpg', img0)[1].tobytes())


def get_system_info():
    def get_temperature():
        try:
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                return temps['coretemp'][0].current
            elif 'cpu-thermal' in temps:
                return temps['cpu-thermal'][0].current
            elif 'acpitz' in temps:
                return temps['acpitz'][0].current
            else:
                return 'N/A'
        except Exception as e:
            logging.error(f"Error reading temperature: {e}")
            return 'N/A'

    temp = get_temperature()
    memory = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=1)
    disk = psutil.disk_usage('/')

    return {
        'temperature': temp,
        'memory': memory.percent,
        'cpu': cpu,
        'disk': disk.percent
    }


@app.route('/')
def index():
    if 'logged_in' in session:
        return render_template('index.html')
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            return 'Invalid credentials', 401
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))


@app.route('/models', methods=['GET'])
def get_models():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    models = [f for f in os.listdir(model_folder) if f.endswith('.onnx')]
    return jsonify(models)


@app.route('/labels', methods=['GET'])
def get_labels():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    labels = [f for f in os.listdir(label_folder) if f.endswith('.json')]
    return jsonify(labels)


@app.route('/label_content', methods=['GET'])
def get_label_content():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    label_name = request.args.get('label')
    label_path = os.path.join(label_folder, label_name)
    with open(label_path, 'r') as f:
        label_content = json.load(f)
    return jsonify(label_content)


@app.route('/change_model', methods=['GET'])
def change_model():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    global model_file_path
    model_name = request.args.get('model')
    model_file_path = os.path.join(model_folder, model_name)
    initialize_yolo_and_camera()
    return jsonify({"message": f"Model changed to {model_name}"}), 200


@app.route('/change_label', methods=['GET'])
def change_label():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    global label_file_path
    label_name = request.args.get('label')
    label_file_path = os.path.join(label_folder, label_name)
    initialize_yolo_and_camera()
    return jsonify({"message": f"Label changed to {label_name}"}), 200


@app.route('/video_feed')
def video_feed():
    if 'logged_in' not in session:
        return redirect(url_for('login'))

    def generate():
        while True:
            frame = frame_queue.get()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/system_info')
def system_info():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    info = get_system_info()
    return jsonify(info)


@app.route('/detections')
def get_detections():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    c.execute("SELECT track_id, label, first_seen, last_seen FROM tracking_records")
    detections = c.fetchall()
    conn.close()
    return jsonify(detections)


@app.route('/detection_stats')
def get_detection_stats():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    c.execute("SELECT label, COUNT(*), AVG((julianday(last_seen) - julianday(first_seen)) * 24 * 60 * 60) FROM tracking_records GROUP BY label")
    stats = c.fetchall()
    conn.close()
    return jsonify(stats)

initialize_yolo_and_camera()

if __name__ == "__main__":
    sleep(1)
    threading.Thread(target=yolo_inference, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)
