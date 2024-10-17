'''
Ymai's Notes
    This appears to be an out of date version of gesture.py that also allows you to pick a version of the model
    I dont understand why this exists
'''

import sys
import os
import cv2
import torch
import mediapipe as mp
import numpy as np
from torchvision import transforms
from PIL import Image
from models.model import SignLanguageCNN  # Ensure this import is correct

def load_model_version(version_number):
    """
    Load a specific version of the trained model.

    Parameters:
        version_number (int): The version number of the model to load.

    Returns:
        model (SignLanguageCNN): The loaded model with weights from the specified version.
    """
    # Construct the path to the checkpoint
    checkpoint_path = os.path.join(
        os.path.dirname(__file__),  # Current directory (scripts)
        '..', 'tb_logs', 'sign_language_mnist_model', f'version_{version_number}', 'checkpoints'
    )
    # Find the first checkpoint file
    checkpoint_file = [f for f in os.listdir(checkpoint_path) if f.endswith('.ckpt')][0]
    # Full path to the checkpoint file
    model_path = os.path.join(checkpoint_path, checkpoint_file)

    # Initialize and load model
    model = SignLanguageCNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))["state_dict"], strict=False)
    model.eval()  # Set to evaluation mode

    return model


def main():
    """
    Main function for real-time gesture recognition using webcam input.
    It loads a trained model, captures video from the webcam, and predicts the sign language gesture in real-time.
    """
    # Load the latest model
    version_to_load = 27  # You can change the version number if necessary
    model = load_model_version(version_to_load)

    # Set up MediaPipe for hand tracking
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

    # Set up the image preprocessing transform
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    # Label mapping
    label_mapping = {
        0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 10: 'K', 11: 'L',
        12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V',
        22: 'W', 23: 'X', 24: 'Y'
    }

    # Start capturing video from the webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame for hand tracking
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract the bounding box for the hand
                x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
                x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
                y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])
                y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])

                # Crop and preprocess the hand image
                hand_crop = frame[y_min:y_max, x_min:x_max]
                if hand_crop.size != 0:
                    hand_crop_pil = Image.fromarray(cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB))
                    hand_tensor = transform(hand_crop_pil).unsqueeze(0)

                    # Make prediction
                    with torch.no_grad():
                        output = model(hand_tensor)
                        _, predicted = torch.max(output, 1)
                        gesture_label = predicted.item()

                        # Display the predicted letter on the screen
                        letter = label_mapping.get(gesture_label, "Unknown")
                        cv2.putText(frame, f'Letter: {letter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Draw landmarks on the hand
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Draw a green bounding box around the detected hand
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Display the webcam frame
        cv2.imshow('Gesture Recognition', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
