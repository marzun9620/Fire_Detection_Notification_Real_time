import torch
import cv2
import cvzone
import numpy as np
from threading import Thread
import signal
import sys
import smtplib
import ssl
from email.message import EmailMessage
import time
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


EMAIL_SENDER = '' //write sender email
EMAIL_PASSWORD = ''  // sender email app password. Get it from your security option after 2FA on.
EMAIL_RECEIVER = ''// receiver email for notification


def create_email_message(confidence):
    subject = 'Fire Detected!'
    body = f"Fire detected with {confidence}% confidence!"
    em = EmailMessage()
    em['From'] = EMAIL_SENDER
    em['To'] = EMAIL_RECEIVER
    em['Subject'] = subject
    em.set_content(body)
    return em

def send_email(confidence):
    message = create_email_message(confidence)
    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, message.as_string())
        logging.info('Email sent successfully!')
    except Exception as e:
        logging.error(f'Failed to send email: {e}')


ESP32_URL = 'http://10.103.133.68:81/stream?fps=15'
cap = cv2.VideoCapture(ESP32_URL)

if not cap.isOpened():
    logging.error("Error: Could not open video stream.")
    sys.exit()
                                                                                                

MODEL_PATH = '' // model path
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)
model.eval()


CLASSNAMES = ['fire']


frame_buffer = None
stop_thread = False
last_email_time = 0
EMAIL_INTERVAL = 300  # 5 minutes

# Set resolution
RESOLUTION = (640, 480)

def capture_frames():
    global frame_buffer, stop_thread
    while not stop_thread:
        ret, frame = cap.read()
        if ret:
            frame_buffer = frame


thread = Thread(target=capture_frames)
thread.start()

def is_fire_color_present(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_fire = np.array([18, 50, 50])
    upper_fire = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower_fire, upper_fire)
    return np.any(mask)

def signal_handler(sig, frame):
    global stop_thread
    stop_thread = True
    thread.join()
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

while True:
    if frame_buffer is None:
        continue

    frame = frame_buffer
    frame_resized = cv2.resize(frame, RESOLUTION)

    if is_fire_color_present(frame_resized):
        rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        results = model(rgb_frame)

        fire_detected = False
        for result in results.xyxy[0].cpu().numpy():
            x1, y1, x2, y2, conf, cls = result
            confidence = int(conf * 100)
            if confidence > 50:
                fire_detected = True
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cvzone.putTextRect(frame_resized, f'{CLASSNAMES[int(cls)]} {confidence}%', [x1 + 8, y1 + 25], scale=1, thickness=1)
                logging.info(f'Detected {CLASSNAMES[int(cls)]} with confidence {confidence}%')

                current_time = time.time()
                if (current_time - last_email_time) >= EMAIL_INTERVAL:
                    send_email(confidence)
                    last_email_time = current_time

    cv2.imshow('Frame', frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stop_thread = True
thread.join()
cap.release()
cv2.destroyAllWindows()