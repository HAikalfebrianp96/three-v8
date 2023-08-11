import cv2
import time
from ultralytics import YOLO
import supervision as sv
import streamlit as st

st.title("Tree Height Detection and Classification")
video_placeholder = st.image([])


def cari_tinggi_asli(h, jarak_treshold):
    hasil = ''
    tinggiAsli = 0
    if (jarak_treshold == 5):
        tinggiAsli = h * 0.826
    elif (jarak_treshold == 10):
        tinggiAsli = h * 1.223
    elif (jarak_treshold == 15):
        tinggiAsli = h * 1.843
    else:
        tinggiAsli = h * 2.439

    if 0 <= tinggiAsli < 100:
        hasil = 'terlalu pendek'
    elif 100 <= tinggiAsli < 200:
        hasil = 'Tinggi Pohon : (1 - 2) meter'
    elif 200 <= tinggiAsli < 300:
        hasil = 'Tinggi Pohon: (2 - 3) meter'
    elif 300 <= tinggiAsli < 400:
        hasil = 'Tinggi Pohon: (3 - 4) meter'
    elif 400 <= tinggiAsli < 500:
        hasil = 'Tinggi Pohon: (4 - 5) meter'
    elif 500 <= tinggiAsli < 600:
        hasil = 'Tinggi Pohon: (5 - 6) meter'
    elif 600 <= tinggiAsli < 700:
        hasil = 'Tinggi Pohon: (6 - 7) meter'
    elif 700 <= tinggiAsli < 800:
        hasil = 'Tinggi Pohon: (7 - 8) meter'
    elif 800 <= tinggiAsli < 900:
        hasil = 'Tinggi Pohon: (8 - 9) meter'
    elif 900 <= tinggiAsli <= 1000:
        hasil = 'Tinggi Pohon: (9 - 10) meter'
    elif 1000 <= tinggiAsli <= 1100:
        hasil = 'Tinggi Pohon: (10 - 11) meter'
    elif 1100 <= tinggiAsli <= 1200:
        hasil = 'Tinggi Pohon: (11 - 12) meter'
    elif 1200 <= tinggiAsli <= 1300:
        hasil = 'Tinggi Pohon: (12 - 13) meter'
    elif 1300 <= tinggiAsli <= 1400:
        hasil = 'Tinggi Pohon: (13 - 14) meter'
    elif 1400 <= tinggiAsli <= 1500:
        hasil = 'Tinggi Pohon: (14 - 15) meter'
    else:
        hasil = 'Terlalu tinggi'
    return hasil


def main(confidence_threshold, jarak_treshold):
    frame_width, frame_height = 1220, 640   # Define the frame resolution
    # Capture video from webcam
    cap = cv2.VideoCapture("DJI_0415.mp4")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    model = YOLO("801010.pt")
    display_width, display_height = 1220, 640  # Define the display size
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.3,
        text_padding=3
    )

    start_time = None
    elapsed_time = st.empty()
    stop_button = st.button("Stop")  # Add a stop button
    placeholder = st.empty()

    while not stop_button:
        ret, frame = cap.read()

        if not ret:
            break

        placeholder.empty()
        result = model(frame)[0]
        detections = sv.Detections.from_yolov8(result)
        detections = detections[detections.confidence > confidence_threshold]

        labels_bawah = [
            f"{cari_tinggi_asli(xy[3]-xy[1],jarak_treshold)}"
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
            # Resizing the frame
            frame = cv2.resize(frame, (display_width, display_height))
        # Update the video frame in the placeholder
        video_placeholder.image(frame, channels="BGR")

        if start_time is None:
            start_time = time.time()

        elapsed = time.time() - start_time
        elapsed_time.markdown(f"Elapsed Time: {elapsed:.2f} seconds")

        with placeholder.container():
            for x in range(len(labels_atas)):
                st.write(f"{labels_atas[x]} dan {labels_bawah[x]}")

        time.sleep(1)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()


if __name__ == "__main__":
    confidence_threshold = st.slider(
        "Confidence Threshold", 0.0, 1.0, 0.7, step=0.1)
    jarak_treshold = st.slider(
        "jarak drone", 5, 20, 10, step=5
    )
    main(confidence_threshold, jarak_treshold)
