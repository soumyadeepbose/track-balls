import cv2

# Displaying mouse poition
def show_mouse_position(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        cv2.setWindowTitle(window_name, f"Mouse Position - x: {x}, y: {y}")

video_path = 'AI Assignment video.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame.")
    cap.release()
    exit()

# Window creation and mouse callback function
window_name = 'Coordinate Detective'
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, show_mouse_position)
frame = cv2.resize(frame, (600, 400))

cv2.imshow(window_name, frame)

cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()