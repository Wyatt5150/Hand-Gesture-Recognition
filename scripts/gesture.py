import os
import sys

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import cv2
import mediapipe as mp
import numpy as np
from torchvision import transforms
from PIL import Image
from models.model import SignLanguageCNN  # Now this should work correctly


def load_version(version):
    """
    Load a specific version of the trained model.

    Parameters:
        version (int): The version number of the model to load.

    Returns:
        model (SignLanguageCNN): The loaded model.
    """
    # Define the path to the model checkpoint
    model_path = os.path.join('C:/Users/Lopez/Documents/Hand-Gesture-Recognition/tb_logs/sign_language_mnist_model/version_0/checkpoints/epoch=29-val_loss=0.00.ckpt')


    # Load the model
    model = SignLanguageCNN.load_from_checkpoint(model_path)
    model.eval()  # Set the model to evaluation mode

    return model



def main():
    """
    Main function to run the gesture recognition model using webcam input.
    """
    # Load the version 0 model
    model = load_version(0)  # Load the model version you want


    # MediaPipe for hand tracking
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

    # Define the transform to preprocess the hand images
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    # Define your label mapping
    label_mapping = {
        0: 'A',
        1: 'B',
        2: 'C',
        3: 'D',
        4: 'E',
        5: 'F',
        6: 'G',
        7: 'H',
        8: 'I',
        10: 'K',
        11: 'L',
        12: 'M',
        13: 'N',
        14: 'O',
        15: 'P',
        16: 'Q',
        17: 'R',
        18: 'S',
        19: 'T',
        20: 'U',
        21: 'V',
        22: 'W',
        23: 'X',
        24: 'Y'
    }

    # Start capturing video
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame with MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Draw hand landmarks and process the detected hand
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
                    # Convert hand_crop (numpy array) to a PIL Image
                    hand_crop_pil = Image.fromarray(cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB))

                    # Preprocess the cropped image
                    hand_tensor = transform(hand_crop_pil)
                    hand_tensor = hand_tensor.unsqueeze(0)

                    # Make prediction
                    with torch.no_grad():
                        output = model(hand_tensor)
                        _, predicted = torch.max(output, 1)
                        gesture_label = predicted.item()

                        # Get the associated letter from the label
                        letter = label_mapping.get(gesture_label, "Unknown")

                        # Display the predicted letter on the frame
                        cv2.putText(frame, f'Letter: {letter}', (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Draw landmarks
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Show the processed frame
        cv2.imshow('Gesture Recognition', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
