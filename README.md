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


## Important Files/Directories
- scripts
   - gesture.py

      Identifies hand gestures in real time. Type q to quit video.

   - dataModule.py

      Loads and prepares data so it can be used for training

   - trainModel.py

      Used to train model.

   - preProcessing.py
   
      Outdated version of dataModule.py

   - VersionGestures.py
   
      Outdated version of gesture.py with model version feature
   
   - loadModel.py

      unused. Loads specified version of model


- utils
   - TODO
