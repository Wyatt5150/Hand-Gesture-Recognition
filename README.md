# Hand-Gesture-Recognition
Project for Computer Vision


## First Time Set Up

Clone Repo from GitBash
1. Create a new directory to pull the project into
2. Open GitBash from that new directory
3. Clone the repository:
    
    git clone https://github.com/Wyatt5150/Hand-Gesture-Recognition.git
4. Open the cloned folder in your IDE (I use VScode)

Install MiniConda
https://docs.anaconda.com/miniconda/

Set Up Environment
1. Open project folder in your IDE
2. Create initial Conda environment, run in terminal:

    conda env create --file environment.yml


## Running The Project

Set your Conda Environment
1. Activate your conda environment:

    conda activate CS4337
2. To update your environment to the latest dependencies:

    conda env update --file environment.yml 
3. To run the project:

    python main.py


## Project File Overview
1. Data Directory
   - CSV Files to Train and Test model
   - Reference ASL PNG


2. Images Directory
   - Testing images set for ASL


3. Models Directory
   - Defines CNN
   - Forward and Backward Pass


4. Scripts Directory
   - Gesture: Real Time Hand Recognition. Type q to quit video.
   - preProcessing: Prepares data before uploads
   - testingCSV: Evaluates CSV data & Randomly shows 5 images
   with associated letter.
   - trainModel: Process for CNN to learn 
   - validateImage: Processes New 3 channel images 


6. tb_logs Directory
   - Tensor Board that tracks Loss, 
   Accuracy, Learning Rate, Gradient Descent


6. utils Directory
   - unused  