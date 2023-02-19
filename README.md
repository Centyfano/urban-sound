# Urban Sound  
This project is a PyTorch implementation of a speech recognition system that can classify audio samples into different categories based on the type of sound they contain. The system is trained and tested on the UrbanSound8K dataset, which consists of 8,732 audio files of different types of sounds found in urban environments, such as car horns, sirens, and street music.  

The system preprocesses the audio files using Mel spectrograms and trains a deep neural network to classify the spectrograms into the different sound categories. The network architecture used is a convolutional neural network (CNN) with multiple layers, followed by a fully connected layer and a softmax activation function.

## Requirements
* Python 3.6 or later
* PyTorch
* pandas

## Installation
* Clone the repository: git clone https://github.com/Centyfano/urban-sound.git
* Install the required dependencies: `pip install -r requirements.txt`
* Download the datsaset [here](https://urbansounddataset.weebly.com) and extract it to `data/`

## Usage
1. Download the UrbanSound8K dataset from [link to dataset]. This dataset is used to train and test the model.
1. Preprocess the dataset using preprocess.py script. This script converts the audio files to Mel spectrograms and saves them to disk.
1. Train the model using train.py script. This script trains the model on the preprocessed dataset and saves the trained model parameters to disk.
1. Evaluate the model using evaluate.py script. This script evaluates the trained model on a test set and prints the evaluation metrics.
1. Use the trained model for inference using infer.py script. This script loads the trained model parameters and uses them to make predictions on new audio files.  


## Credits
This project uses the UrbanSound8K dataset, which is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license. The dataset was created by Justin Salamon and Juan Pablo Bello at the Music and Audio Research Laboratory (MARL) at New York University. For more information, see the dataset website: https://urbansounddataset.weebly.com/.  

The project is based on a tutorial by **Valerio Velardo - The Sound of AI** on building a speech recognition system using PyTorch. The tutorial was used as a reference for the implementation of the neural network architecture and training process. For more information, see the tutorial video: https://youtube.com/playlist?list=PL-wATfeyAMNoirN4idjev6aRu8ISZYVWm.

