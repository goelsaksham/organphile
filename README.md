# organphile
Repository hosting code for my Introduction to Artificial Intelligence Project - Organphile (An Instrument Recognizer)

## Project Overview
This is a project that corresponds to using different deep learning architectures for identifying which instrument is
 being played in a particular audio excerpt. The goal of the project is to do real time classification for the 
 instrument being played in a particular audio excerpt whether it is being heard from a microphone or is a audio 
 file which has been recorded earlier. Currently for the project we are using IRMAS Dataset and more information 
 about the dataset can be found in the following [section](#abcd). 


## <a name="abcd">Dataset</a>


## Project Timeline


## Project Components


## Project Resources


## Project Updates


## Project TODO:
 * Create a Training Data Loader Script
   * Add functions to load training data from the .wav files
   * Zero Padding
   * Categorical variables to Numerical
   * Sliding Window on top of the original loaded data to divide it into the chunks of times 
 * Training Data Augmentation (Model Based)
   * Sliding Window on top of each data sample to convert it to a vector and time based 2-D array model
 * Create a Test Data Loader Script
    * Load the data from the .wav files in the test directory and attach the given class labels for that audio


## Project Results


    
## Results

This section holds the results of the various experiments conducted during the course of the project and record all 
the hyperparameters for the type of training along with the model architecture so that these can be added to the 
final report for the AI project.

Date|Notebook|Model|Accuracy|Precision|Recall|Epochs|Time Window|Feature Window|Notes
----|--------|-----|--------|---------|------|------|-----------|--------------|-----
11-13-2018|Vinall Binary RNN Model|Simple RNN|77.47%|-|-|20|1 sec only|147 X 300 - 300 steps| N/A