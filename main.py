import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

input_dir = 'dog.mp4'
cap = cv2.VideoCapture(input_dir)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (720, 486), interpolation=cv2.INTER_AREA)
    
    result = model.track(frame, persist=True)

    frame_ = result[0].plot()


    cv2.imshow('frame', frame_)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()