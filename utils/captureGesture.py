import cv2
import os
from PIL import Image
import torchvision.transforms as transforms

# Define your transformations (you can tweak this as needed)
transform = transforms.Compose([
    transforms.RandomRotation(degrees=15),      # Randomly rotate the image by up to 15 degrees
    transforms.RandomHorizontalFlip(p=0.5),     # Randomly flip the image horizontally
    transforms.RandomVerticalFlip(p=0.5),       # Randomly flip the image vertically
    transforms.RandomResizedCrop(size=200),     # Randomly crop the image and resize to 200x200
    transforms.ToTensor()                       # Convert image to PyTorch tensor (can remove if not needed)
])

def capture_image(frame):
    """
    Save the captured frame as an image file in the designated directory.

    Parameters:
        frame (numpy.ndarray): The image frame captured from the webcam.

    Returns:
        None
    """
    global capture_count
    capture_count += 1

    # Convert OpenCV image (NumPy array) to PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Apply transformations to the image
    augmented_image = transform(pil_image)

    # Convert back to PIL format to save as an image file
    augmented_image = transforms.ToPILImage()(augmented_image)

    image_path = os.path.join(image_dir, f'{gesture_label}_{capture_count}.png')
    augmented_image.save(image_path)
    print(f'Captured and augmented image {capture_count} at {image_path}')


# Get the gesture label from the user
gesture_label = input("Enter the gesture label (e.g., A, B, C): ").upper()

# Set up directories for saving images
image_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images', gesture_label)

if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# Initialize OpenCV and hand detection variables
cap = cv2.VideoCapture(0)
capture_count = 0
capture_limit = 1500  # You can change this to the number of images you want to capture
green_box_position = (500, 300, 500, 500)  # Coordinates for the green box (x, y, width, height)

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
