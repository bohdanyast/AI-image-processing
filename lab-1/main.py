import cv2
import numpy as np
import imutils

print("Welcome to Console CV2 App!")
print(
"""
Options!
1 - display image
2 - save image
3 - get random pixel info
4 - get random Region of Interests
5 - change image size
6 - rotate image
7 - blur image
8 - paint line
9 - paint polylines
10 - paint circle
11 - put text
"""
)

image_file = input("Paste Image Filepath: ")
image_file_flag = int(input(
"""
Choose channel:
1 - colors, no alpha
0 - grayscale
-1 - colors + alpha
"""))

option = int(input("Enter option number: "))


def display_image(filename, flag=cv2.IMREAD_COLOR):
    img = cv2.imread(filename, flag)
    cv2.imshow("Option 1: Displaying", img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def save_image(filename, flag=cv2.IMREAD_COLOR):
    dest = input("Write destination folder with desired filename: ")
    filename_cv2 = cv2.imread(filename, flag)
    cv2.imwrite(dest, filename_cv2)


def get_random_pixel_info(filename, flag=cv2.IMREAD_COLOR):
    img = cv2.imread(filename, flag)

    dimensions = img.shape
    h, w = dimensions[0], dimensions[1]

    x = np.random.randint(0, w)
    y = np.random.randint(0, h)

    if len(dimensions) == 3:
        # BGR format
        b, g, r = img[y, x]
        print(f"Pixel at ({x}, {y}) -> R:{r}, G:{g}, B:{b}")
    else:
        # Grayscale format
        intensity = img[y, x]
        print(f"Pixel at ({x}, {y}) -> Intensity:{intensity}")


def get_random_roi(filename, flag=cv2.IMREAD_COLOR):
    img = cv2.imread(filename, flag)

    h, w = img.shape[:2]

    p1 = (np.random.randint(0, w), np.random.randint(0, h))
    p2 = (np.random.randint(0, w), np.random.randint(0, h))

    x1, x2 = sorted([p1[0], p2[0]])
    y1, y2 = sorted([p1[1], p2[1]])

    roi = img[y1:y2, x1:x2]

    cv2.imshow("Option 4: ROI", roi)
    cv2.waitKey()
    cv2.destroyAllWindows()


def change_image_size(filename, flag, new_h, new_w):
    option_proportion = int(input("Enter 0 - for exact size, 1 - for proportional: "))

    img = cv2.imread(filename, flag)

    if option_proportion == 0:
        resized = cv2.resize(img, (new_h, new_w))
    else:
        resized = imutils.resize(img, height=new_h, width=new_w)

    cv2.imshow("Option 5: New Image Size", resized)
    cv2.waitKey()
    cv2.destroyAllWindows()


def rotate_image(filename, flag, deg):
    img = cv2.imread(filename, flag)
    rotated = imutils.rotate(img, deg)

    cv2.imshow("Option 6: Rotate Image", rotated)
    cv2.waitKey()
    cv2.destroyAllWindows()


def blur_image(filename, flag):
    img = cv2.imread(filename, flag)
    blurred = cv2.GaussianBlur(img, (11, 11), 0)

    cv2.imshow("Option 7: Blur Image", blurred)
    cv2.waitKey()
    cv2.destroyAllWindows()


def paint_line():
    img = np.zeros((200, 200, 3), np.uint8)
    cv2.line(img, (0, 0), (100, 100), (123, 123, 123), 4)

    cv2.imshow("Option 8: Line", img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def paint_poly_lines():
    img = np.zeros((200, 200, 3), np.uint8)
    points = np.array([[12, 12], [123, 43], [43, 123]])
    cv2.polylines(img, [points], isClosed=True, color=(100, 23, 12), thickness=2)

    cv2.imshow("Option 9: Polyline", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def paint_circle():
    img = np.zeros((200, 200, 3), np.uint8)
    cv2.circle(img, (100, 100), 75, (0, 0, 255), 5)

    cv2.imshow("Option 10: Circle", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def put_text(text):
    img = np.zeros((200, 200, 3), np.uint8)
    font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    cv2.putText(
        img, text, (30, 100), font, 4, (0, 0, 255), 5
    )

    cv2.imshow("Option 11: Text", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


match option:
    case 1:
        display_image(image_file, image_file_flag)
    case 2:
        save_image(image_file, image_file_flag)
    case 3:
        get_random_pixel_info(image_file, image_file_flag)
    case 4:
        get_random_roi(image_file, image_file_flag)
    case 5:
        new_h = int(input("Enter new height: "))
        new_w = int(input("Enter new width: "))
        change_image_size(image_file, image_file_flag, new_h, new_w)
    case 6:
        deg = int(input("Enter degrees to rotate: "))
        rotate_image(image_file, image_file_flag, deg)
    case 7:
        blur_image(image_file, image_file_flag)
    case 8:
        paint_line()
    case 9:
        paint_poly_lines()
    case 10:
        paint_circle()
    case 11:
        text = input("Enter text to display: ")
        put_text(text)