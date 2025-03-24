# this is the code for the ML project spring 2025
# the purpose of code is to take video input (video file) and get a ML database to detect what is the video.

from ultralytics import YOLO
import cv2

# Load a pre-trained YOLOv8 model (trained on COCO)
model = YOLO("yolov8n.pt")  # You can also try yolov8s.pt or yolov8m.pt for better accuracy

# Load your video file
video_path = "RingVideo1.mp4"  # Replace this with your actual file name
cap = cv2.VideoCapture(video_path)

# Optional: Save the output video
out = None
save_output = True  # Set to False if you don't want to save
if save_output:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter("output_video.mp4", fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection on the frame
    results = model(frame)

    # Plot the results on the frame
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow("Object Detection", annotated_frame)

    # Write to file if saving
    if save_output:
        out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
if save_output:
    out.release()
cv2.destroyAllWindows()
