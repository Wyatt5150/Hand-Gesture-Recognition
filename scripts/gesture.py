'''
Ymai's notes:
    Added a comment in the imports because I made that mistake and will likely do it again without the comment

'''
import sys
import os
import torch
import cv2
import mediapipe as mp
import numpy as np
from torchvision import transforms
from PIL import Image
import time

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# DONT MOVE THESE ABOVE sys.path.append(...) OR ELSE THE PROGRAM CANT FIND THEM
from models.model import SignLanguageCNN  # Ensure this import is correct
from scripts.stabilizer import Stabilizer

# Define the label mapping (0-23 -> A-Z excluding J and Z)
'''
label_mapping : dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P',
    15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W',
    22: 'X', 23: 'Y'
}'''
label_mapping:dict = {}
file = open("validLetters.txt", 'r+')
i = 0
for letter in file.read().upper().split():
    label_mapping[i] = letter
    i+=1
file.close()
print(label_mapping)

def load_model() -> SignLanguageCNN:
    """
    Loads the trained SignLanguageCNN model from a checkpoint file.

    Returns:
        SignLanguageCNN : The loaded and initialized gesture recognition model in evaluation mode.
    """
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pth')
    model = SignLanguageCNN(num_classes = len(label_mapping.keys()))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
    model.eval()
    return model


def preprocess_image(hand_crop : np.ndarray) -> torch.Tensor:
    """
    Preprocesses a cropped hand image to prepare it for input to the model.
    Converts the image to grayscale, resizes it to 28x28 pixels, and transforms it into a tensor.

    Parameters:
        hand_crop (numpy.ndarray): The cropped image of the hand, as a NumPy array from OpenCV.

    Returns:
        torch.Tensor: The preprocessed image as a 4D tensor suitable for model input (batch size, channels, height, width).
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    # Convert hand_crop (numpy array) to a PIL Image
    hand_crop_pil = Image.fromarray(cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB))
    hand_tensor = transform(hand_crop_pil)
    hand_tensor = hand_tensor.unsqueeze(0)  # Add batch dimension
    return hand_tensor


def main():
    """
    Main function to run the gesture recognition model on webcam input in real-time.

    Uses MediaPipe for hand tracking and captures live video feed to detect and classify hand gestures.
    Draws hand landmarks, crops the hand region, and predicts the gesture using the loaded model.
    Displays the predicted letter on the screen and terminates on pressing the 'q' key.
    """
    # Load the model
    model = load_model()

    # MediaPipe for hand tracking
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

    # Start capturing video
    cap = cv2.VideoCapture(0)

    # Where to split image in half
    ret, frame = cap.read()
    width_center = frame.shape[1]//2

    # black block to visually divide left and right frames
    block= np.zeros((frame.shape[0], 10, 3), dtype=np.uint8)

    # Stabilizer
    stabilizers = [Stabilizer(),Stabilizer()]

    while cap.isOpened():
        ret, videoFrame = cap.read()
        if not ret:
            break
        videoFrame = cv2.flip(videoFrame, 1)
        # Process the frame with MediaPipe
        #frame_rgb = cv2.cvtColor(videoFrame, cv2.COLOR_BGR2RGB)

        # split camera fead into halves
        frames = [videoFrame[:, :width_center], videoFrame[:, width_center:]]

        # Draw hand landmarks and process the detected hand
        for i in range(2):
            frame = frames[i]
            stabilizer = stabilizers[i]
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract the bounding box for the hand
                    x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
                    x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
                    y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])
                    y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])

                    # Crop the hand region from the frame
                    hand_crop = frame[y_min:y_max, x_min:x_max]
                    if hand_crop.size != 0:
                        hand_tensor = preprocess_image(hand_crop)

                        # Make prediction
                        with torch.no_grad():
                            output = model(hand_tensor)

                            # Log the raw output values for debugging
                            print("Model output:", output)

                            confidence, predicted = torch.max(output, 1)
                            gesture_label = predicted.item()


                            # Get the associated letter from the label
                            letter = label_mapping.get(gesture_label, "Unknown")
                            letter = stabilizer.group_stabilize(letter, confidence)

                            # Display the predicted letter on the frame
                            cv2.putText(frame, f'Letter: {letter}', (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Draw landmarks
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Show the processed frame
        cv2.imshow('Gesture Recognition', cv2.hconcat([frames[0],block,frames[1]]))
        # reduce how often the loop runs so my laptop doesnt explode
        time.sleep(.05)
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
