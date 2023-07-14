import cv2
from ultralytics import YOLO
import supervision as sv
import streamlit as st

st.title("Look Up at the Stars")
video_placeholder = st.image([])
def cari_tinggi_asli(h, w):
    hasil = ''
    real = 4
    nilaiPerbandingan = real / min(h, w)
    tinggiAsli = h * nilaiPerbandingan
    if 1 <= tinggiAsli < 2:
        hasil = 'Height: (1 - 2) meters'
    elif 3 <= tinggiAsli < 4:
        hasil = 'Height: (3 - 4) meters'
    elif 4 <= tinggiAsli < 5:
        hasil = 'Height: (4 - 5) meters'
    elif 5 <= tinggiAsli < 6:
        hasil = 'Height: (5 - 6) meters'
    elif 6 <= tinggiAsli < 7:
        hasil = 'Height: (6 - 7) meters'
    elif 7 <= tinggiAsli < 8:
        hasil = 'Height: (7 - 8) meters'
    elif 8 <= tinggiAsli < 9:
        hasil = 'Height: (8 - 9) meters'
    elif 9 <= tinggiAsli <= 10:
        hasil = 'Height: (9 - 10) meters'
    else:
        hasil = 'Too high'
    return hasil
def main():
    frame_width, frame_height = 1280, 720  # Define the frame resolution
    cap = cv2.VideoCapture("yy.mp4")  # Capture video from webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    model = YOLO("1000.pt")
    display_width, display_height = 1220, 640  # Define the display size
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.3,
        text_padding=3
    )

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        result = model(frame)[0]
        detections = sv.Detections.from_yolov8(result)
        detections = detections[detections.confidence > 0.70]

        labels_bawah = [
            f"{cari_tinggi_asli(xy[3] - xy[1], xy[2] - xy[0])}"
            for xy, confidence, class_id, _
            in detections
        ]
        labels_atas = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for xy, confidence, class_id, _
            in detections
        ]
        frame = box_annotator.annotate(
            scene=frame, detections=detections, labels_atas=labels_atas, labels_bawah=labels_bawah
        )
        if frame is not None:
            frame = cv2.resize(frame, (display_width, display_height))  # Resizing the frame
        video_placeholder.image(frame, channels="BGR")  # Update the video frame in the placeholder
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
if __name__ == "__main__":
    main()
