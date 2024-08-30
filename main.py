import cv2 as cv
import tensorflow as tf
import pandas as pd

def main():
    print("TensorFlow version:", tf.__version__)
    print("OpenCV version:", cv.__version__)
    print("Pandas version:", pd.__version__)

if __name__ == "__main__":
    main()