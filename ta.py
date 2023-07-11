import cv2
import argparse

from ultralytics import YOLO
import supervision as sv
import numpy as np


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280, 720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args


def cari_tinggi_asli(h, w):
    hasil = ''

    nilaiPerbandingan = 4/min(h, w)
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
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture("videoframe.mp4")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("1000.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.3,
        text_padding=3
    )

    while True:
        ret, frame = cap.read()
        tinggi = 10
        result = model(frame)[0]
        detections = sv.Detections.from_yolov8(result)
        detections = detections[detections.confidence > 0.5]

        labels_bawah = [
            f"{cari_tinggi_asli(xy[3]-xy[1],xy[2]-xy[0])}"
            for xy, confidence, class_id, _
            in detections
        ]
        labels_atas = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for xy, confidence, class_id, _
            in detections
        ]
        frame = box_annotator.annotate(
            scene=frame,  detections=detections, labels_atas=labels_atas, labels_bawah=labels_bawah)

        cv2.imshow("yolov8", frame)

        if (cv2.waitKey(30) == 27):
            break


if __name__ == "__main__":
    main()
