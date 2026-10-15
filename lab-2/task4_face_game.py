import cv2
import time
import random

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

vid = cv2.VideoCapture(0)

# Width of app and zone. Zone coords
W, H = 800, 600

ZONE_W = W // 4
ZONE_H = H // 4

ZONES = {
    "TOP": (W//2 - ZONE_W//2, 0, ZONE_W, ZONE_H),
    "BOTTOM": (W//2 - ZONE_W//2, H - ZONE_H, ZONE_W, ZONE_H),
    "LEFT": (0, H//2 - ZONE_H//2, ZONE_W, ZONE_H),
    "RIGHT": (W - ZONE_W, H//2 - ZONE_H//2, ZONE_W, ZONE_H),
}


def pick_zone():
    name = random.choice(list(ZONES.keys()))
    return name, ZONES[name]


def face_in_zone(fx, fy, fw, fh, zx, zy, zw, zh):
    face_cx = fx + fw // 2
    face_cy = fy + fh // 2
    return zx <= face_cx <= zx + zw and zy <= face_cy <= zy + zh


def draw_zone(frame, zx, zy, zw, zh, color, label):
    cv2.rectangle(frame, (zx, zy), (zx + zw, zy + zh), color, 4)
    cv2.putText(frame, label, (zx + 8, zy + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


# Game stats
score_ok = 0
score_fail = 0

zone_name, (zx, zy, zw, zh) = pick_zone()
zone_start = time.time()
TIMEOUT = 6.0

result_text = ""
result_color = (255, 255, 255)
result_until = 0.0

while True:
    ret, frame = vid.read()
    if not ret:
        break

    frame = cv2.resize(frame, (W, H))
    frame = cv2.flip(frame, 1)  # Flipping the image to avoid mirror
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    now = time.time()
    elapsed = now - zone_start
    left = max(0.0, TIMEOUT - elapsed)

    # Color for time left
    if left > 1.5:
        zone_color = (0, 220, 255)
    elif left > 3.0:
        zone_color = (0, 140, 255)
    else:
        zone_color = (0, 0, 255)

    draw_zone(frame, zx, zy, zw, zh, zone_color, zone_name)

    # Timer
    cv2.putText(frame, f"{left:.1f}s", (zx + zw // 2 - 20, zy + zh // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, zone_color, 2)

    # Face
    hit = False
    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 200, 0), 2)

        face_cx = fx + fw // 2
        face_cy = fy + fh // 2
        cv2.circle(frame, (face_cx, face_cy), 6, (0, 255, 255), -1)  # заповнене коло

        if face_in_zone(fx, fy, fw, fh, zx, zy, zw, zh):
            hit = True

    # Check if fail or pass
    if hit:
        score_ok += 1
        result_text = "IN ZONE!"
        result_color = (0, 220, 80)
        result_until = now + 0.8
        zone_name, (zx, zy, zw, zh) = pick_zone()
        zone_start = now

    elif elapsed >= TIMEOUT:
        score_fail += 1
        result_text = "NOT IN TIME!"
        result_color = (0, 60, 220)
        result_until = now + 0.8
        zone_name, (zx, zy, zw, zh) = pick_zone()
        zone_start = now

    # Result of prev round
    if now < result_until:
        cv2.putText(frame, result_text, (W // 2 - 90, H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, result_color, 3)

    # Tableau
    cv2.putText(frame, f"OK: {score_ok}   FAIL: {score_fail}", (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Face Zone Game", frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

print(f"Влучань: {score_ok}")
print(f"Промахів: {score_fail}")
print(f"Точність: {score_ok / (score_ok + score_fail) * 100:.1f}%")

vid.release()
cv2.destroyAllWindows()
