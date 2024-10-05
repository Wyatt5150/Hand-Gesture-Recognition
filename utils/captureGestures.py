import cv2
import os

# Get the gesture label from the user
gesture_label = input("Enter the gesture label (e.g., A, B, C): ").upper()

# Set up directories for saving images
image_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images', gesture_label)

if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# Initialize OpenCV and hand detection variables
cap = cv2.VideoCapture(0)
capture_count = 0
capture_limit = 500  # You can change this to the number of images you want to capture
green_box_position = (50, 50, 200, 200)  # Coordinates for the green box (x, y, width, height)

def capture_image(frame):
    global capture_count
    capture_count += 1
    image_path = os.path.join(image_dir, f'{gesture_label}_{capture_count}.png')
    cv2.imwrite(image_path, frame)
    print(f'Captured image {capture_count} at {image_path}')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw the green box on the frame
    x, y, w, h = green_box_position
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the number of images captured
    cv2.putText(frame, f'Captured: {capture_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Capture Hand Gesture', frame)

    # Press 'Enter' to capture an image
    if cv2.waitKey(1) & 0xFF == 13:  # Enter key
        # Capture the hand inside the green box area
        hand_crop = frame[y:y + h, x:x + w]
        capture_image(hand_crop)

    # Stop capturing after reaching the limit
    if capture_count >= capture_limit:
        print("Capture limit reached. Exiting...")
        break

    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
