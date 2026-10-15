import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

vid = cv2.VideoCapture('images/pedestrians.mp4')

while True:
    ret, frame = vid.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800, 600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # пішоходи
    boxes, weights = hog.detectMultiScale(gray, winStride=(8, 8), padding=(4, 4), scale=1.05)
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # обличчя всередині пішохода (ROI)
        roi_gray = gray[y1:y2, x1:x2]
        roi_color = frame[y1:y2, x1:x2]

        if roi_gray.size > 0:
            faces = face_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=4)
            for (fx, fy, fw, fh) in faces:
                cv2.rectangle(roi_color, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)

    cv2.putText(frame, f"Pedestrians: {len(boxes)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Task 3", frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

vid.release()
cv2.destroyAllWindows()
