import cv2
from ultralytics import YOLO

# 1. Load your fine-tuned model (Ensure best.pt is in the same folder as this script)
model = YOLO("best.pt")

# 2. Open the local webcam (0 is usually the built-in camera)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit the camera feed.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Run YOLOv8 inference on the live frame
    # We use stream=True for better real-time performance
    results = model.predict(frame, conf=0.5, stream=True)

    for result in results:
        # Plot the bounding boxes and labels on the frame
        annotated_frame = result.plot()

        # 4. Display the resulting frame
        cv2.imshow("YOLOv8 Real-Time License Plate Detection", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()