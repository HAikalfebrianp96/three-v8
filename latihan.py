import cv2
import argparse
import numpy as np
from ultralytics import YOLO
import supervision as sv

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

        if not ret:
            break

        result = model(frame)[0]
        detections = sv.Detections.from_yolov8(result)

        boxes = np.array([detection[0] for detection in detections])
        confidences = np.array([detection[1] for detection in detections])
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

        for i in indices:
            x1, y1, w, h = boxes[i]

            labels = [
                f"{model.model.names[class_id]} {confidence:.2f}"
                for _, confidence, class_id, _
                in detections
            ]

            frame = box_annotator.annotate(
                scene=frame,
                detections=detections,
                labels=labels
            )

            length_pixels = min(w, h)
            height_meters = 4
            conversion_factor = height_meters / length_pixels
            focal_length = h * conversion_factor

            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 2)
            cv2.putText(frame, str(focal_length), (x1, y1 + 60), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 3)
            cv2.putText(frame, str(length_pixels), (x1, y1 + 60), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 3)
            cv2.putText(frame, str(h), (x1, y1 + 100), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 3)
            cv2.putText(frame, str(w), (x1, y1 + 140), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 3)

            conf = confidences[i]
            text = labels[i] + "{:.2f}".format(conf)

            cv2.putText(frame, text, (x1, y1 + 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 3)

            if 1 <= focal_length < 2:
                cv2.putText(frame, 'Height: (1 - 2) meters', (x1, y1 + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif 3 <= focal_length < 4:
                cv2.putText(frame, 'Height: (3 - 4) meters', (x1, y1 + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif 4 <= focal_length < 5:
                cv2.putText(frame, 'Height: (4 - 5) meters', (x1, y1 + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif 5 <= focal_length < 6:
                cv2.putText(frame, 'Height: (5 - 6) meters', (x1, y1 + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif 6 <= focal_length < 7:
                cv2.putText(frame, 'Height: (6 - 7) meters', (x1, y1 + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif 7 <= focal_length < 8:
                cv2.putText(frame, 'Height: (7 - 8) meters', (x1, y1 + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif 8 <= focal_length < 9:
                cv2.putText(frame, 'Height: (8 - 9) meters', (x1, y1 + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif 9 <= focal_length <= 10:
                cv2.putText(frame, 'Height: (9 - 10) meters', (x1, y1 + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'Too high', (x1, y1 + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Convert the image from BGR to RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            cv2.imshow("yolov8", img_rgb)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
