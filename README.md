# asl_recognizer
This project is an ASL recognition system that can interpret images of hand positions and classify them as letters. We will be using the dataset ASL Fingerspelling Images (RGB & Depth), found on Kaggle. The dataset can be found here: https://www.kaggle.com/datasets/mrgeislinger/asl-rgb-depth-fingerspelling-spelling-it-out?resource=download-directory. 

## How to Run
### To Run the CNN
To train and test the CNN, use the train_evaluate_CNN.py file. This can be called from the command line, and various parameters can be set by editing the parameters of the run_main() function. For example, "python3 train_evaluate_CNN.py". This will automatically generate relevant figures such as accuracy and loss plots.

### To Run the Autoencoder and Combined Autoencoder/CNN Model
To train and test the Autoencoder, use the train_evaluate_AutoEncoder.py file. This can be called from the command line, and various parameters can be set by editing the parameters of the run_main() function. For example, "python3 train_evaluate_AutoEncoder.py". This will automatically generate relevant figures such as accuracy and loss plots.

## Required Libraries
The required libraries for this project are PyTorch, NumPy, and Matplotlib. 
