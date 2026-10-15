import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

frame = cv2.imread("images/FrontalFaces.png")

smile_count = 0
eye_count = 0

gray_filter = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray_filter, scaleFactor=1.1, minNeighbors=5)


for i, (x, y, w, h) in enumerate(faces):
    roi_gray = gray_filter[y:y + h, x:x + w]
    roi_color = frame[y:y + h, x:x + w]

    smile = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=20)
    eye = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)

    color = (0, 255, 0) if len(smile) > 0 else (255, 0, 0)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

    cv2.putText(frame, f"Face {i + 1}", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    for (sx, sy, sw, sh) in smile:
        cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 1)
    for (ex, ey, ew, eh) in eye:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 1)

    if len(smile) > 0:
        smile_count += 1
    if len(eye) > 0:
        eye_count += 1

print(f"Осіб знайдено: {len(faces)}")
print(f"З усмішкою: {smile_count}")
print(f"З видимими очима: {eye_count}")

cv2.imwrite("images/result.jpg", frame)
cv2.imshow(f"Faces: {len(faces)}", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()