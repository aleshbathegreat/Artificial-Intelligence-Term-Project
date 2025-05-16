# Optimising a Music Emotion Recogniser and Classifier

This project focuses on the optimisation of a system that recognises and classifies emotions in music using various audio features, neural models, and fuzzy logic.
## Project Structure

- **`GuiWindow.py`**  
  Main driver script for the application. It features a GUI with labelled buttons for the following functionalities:  
  - Play / Pause  
  - Select Feature Extractor / Optimiser  
  - Open Visualiser Window  
  - Generate Emotion Output  
  - Reset Interface

- **`CSVCreation.py`**  
  Script used to generate CSV datasets, located in the `csv_files/` directory.

- **`savedmoduleweights/`**  
  Directory containing pre-trained model weights used by `NeuralOptimiser.py`.

- **`NeuralOptimiser.py`**  
  Implements a basic feedforward neural network (FNN). The model is designed to work with generalised paths to load pre-trained weights.

- **`colab-training/`**  
  Contains all model training scripts. Prediction output files from this folder were used to evaluate accuracy. 

- **`FeatureExtraction.py`**  
  Extracts and standardises audio features, returning the relevant feature vectors for analysis.

- **`EmotionLogic.py`**  
  Applies fuzzy logic and rule-based inference to generate a final crisp emotion classification output.

