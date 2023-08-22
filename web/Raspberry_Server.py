import cv2
import numpy as np
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app)

camera = cv2.VideoCapture(0)  # 카메라 장치 번호 (일반적으로 0)

count_value = 0  # 초기 값 설정
count_a = 0  # 'a' 카운트 변수
count_b = 0  # 'b' 카운트 변수
count_c = 0  # 'c' 카운트 변수
count_d = 0  # 'd' 카운트 변수
count_e = 0 # 'e' 카운트 변수

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html', count=count_value, count_a=count_a, count_b=count_b, count_c=count_c, count_d=count_d, count_e=count_e)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def handle_connect():
    emit('update_count', {'count': count_value, 'count_a': count_a, 'count_b': count_b, 'count_c': count_c, 'count_d': count_d, 'count_e': count_e})

@socketio.on('update_count_from_client')
def handle_update_count(data):
    global count_value, count_a, count_b, count_c, count_d, count_e
    product_name = data['product_name']
    
    product_state = data['product_state']
    if product_name in ['a', 'b', 'c', 'd', 'e']:  
        if product_state == 'in':
            count_value += 1
            if product_name == 'a':
                count_a += 1
            elif product_name == 'b':
                count_b += 1
            elif product_name == 'c':
                count_c += 1
            elif product_name == 'd':
                count_d += 1
            elif product_name == 'e':
                count_e += 1
        elif product_state == 'out':
            if count_value > 0:
                count_value -= 1
            if product_name == 'a':
                if count_a > 0:
                    count_a -= 1
            elif product_name == 'b':
                if count_b > 0:
                    count_b -= 1
            elif product_name == 'c':
                if count_c > 0:
                    count_c -= 1
            elif product_name == 'd':
                if count_d > 0:
                    count_d -= 1
            elif product_name == 'e':
                if count_e > 0:
                    count_e -= 1

        emit('update_count', {'count': count_value, 'count_a': count_a, 'count_b': count_b, 'count_c': count_c, 'count_d': count_d, 'count_e': count_e}, broadcast=True)



@socketio.on('reset_counts_request')
def handle_reset_counts_request():
    global count_value, count_a, count_b, count_c, count_d, count_e
    count_value = 0

    count_a = 0
    count_b = 0
    count_c = 0
    count_d = 0
    count_e = 0
    emit('update_count', {'count_value': count_value, 'count_a': count_a, 'count_b': count_b, 'count_c': count_c, 'count_d': count_d, 'count_e': count_e}, broadcast=True)


if __name__ == '__main__':
    socketio.run(app, host='10.10.14.14', port=8000)
