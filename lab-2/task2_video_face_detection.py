import cv2
import time

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

vid = cv2.VideoCapture(0)

# Session start
session_start = time.time()

face_start = None
smile_start = None
eyes_start = None

face_total = 0.0
smile_total = 0.0
eyes_total = 0.0

max_faces = 0
total_frames = 0


def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def draw_stats(frame, elapsed):
    # Counting total stats
    now = time.time()
    ft = face_total + (now - face_start if face_start else 0)
    st = smile_total + (now - smile_start if smile_start else 0)
    et = eyes_total + (now - eyes_start if eyes_start else 0)

    face_pct = ft / elapsed * 100 if elapsed > 0 else 0
    smile_pct = st / elapsed * 100 if elapsed > 0 else 0

    lines = [
        f"Session time: {format_time(elapsed)}",
        f"Face: {format_time(ft)}  ({face_pct:.1f}%)",
        f"Smile: {format_time(st)}  ({smile_pct:.1f}%)",
        f"Eyes seen: {format_time(et)}",
        f"Max faces: {max_faces}",
        f"Frames: {total_frames}",
    ]

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (310, 22 + len(lines) * 22), (0, 0, 0), -1)

    for i, line in enumerate(lines):
        cv2.putText(frame, line, (16, 30 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


while True:
    ret, img = vid.read()
    if not ret:
        break

    now = time.time()
    elapsed = now - session_start
    total_frames += 1

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    face_count = len(faces)
    max_faces = max(max_faces, face_count)
    smile_in_frame = False
    eyes_in_frame = False

    # Face tracking
    if face_count > 0:
        if face_start is None:
            face_start = now
    else:
        if face_start is not None:
            face_total += now - face_start
            face_start = None

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        smile = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=15)
        eye = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)

        if len(smile) > 0:
            smile_in_frame = True
        if len(eye) > 0:
            eyes_in_frame = True

        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 1)

        for (ex, ey, ew, eh) in eye:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 1)

    # Smile tracking
    if smile_in_frame:
        if smile_start is None:
            smile_start = now
    else:
        if smile_start is not None:
            smile_total += now - smile_start
            smile_start = None

    # Eye tracking
    if eyes_in_frame:
        if eyes_start is None:
            eyes_start = now
    else:
        if eyes_start is not None:
            eyes_total += now - eyes_start
            eyes_start = None

    draw_stats(img, elapsed)
    cv2.imshow("img", img)

    if cv2.waitKey(30) & 0xFF == 27:
        break

end = time.time()
if face_start:
    face_total += end - face_start
if smile_start:
    smile_total += end - smile_start
if eyes_start:
    eyes_total += end - eyes_start

elapsed = end - session_start

print("\nTOTAL: ")
print(f"Тривалість: {format_time(elapsed)}")
print(f"Кадрів оброблено: {total_frames}")
print(f"Обличчя у кадрі: {format_time(face_total)}  ({face_total / elapsed * 100:.1f}%)")
print(f"Усмішка: {format_time(smile_total)}  ({smile_total / elapsed * 100:.1f}%)")
print(f"Очі видно: {format_time(eyes_total)}  ({eyes_total / elapsed * 100:.1f}%)")
print(f"Максимум облич одразу: {max_faces}")

vid.release()
cv2.destroyAllWindows()
