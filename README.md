# organphile
Repository hosting code for my Introduction to Artificial Intelligence Project - Organphile (An Instrument Recognizer)

## Project Overview
This is a project that corresponds to using different deep learning architectures for identifying which instrument is
 being played in a particular audio excerpt. The goal of the project is to do real time classification for the 
 instrument being played in a particular audio excerpt whether it is being heard from a microphone or is a audio 
 file which has been recorded earlier. Currently for the project we are using IRMAS Dataset and more information 
 about the dataset can be found in the following [section](#dataset). 


## <a name="dataset">Dataset</a>
As mentioned earlier we will be using the IRMAS Dataset for finding a solution to our problem. 
The IRMAS dataset currently has divided the dataset into Training and Test sets. 
Training Dataset contains 6705 16 bit stereo wav format audio files sampled at 44.1 kHz. 
All of these files are 3 second excerpts from more than 2000 distinct recordings. 

The number of audio files for each given class is given in Table below:

|Audio File Class|Number of Audio Files|
|----------------|---------------------|
|Cello|388|
|Clarinet|505|
|Flute|451|
|Acoustic Guitar|637|
|Electric Guitar|760|
|Organ|682|
|Piano|721|
|Saxophone|626|
|Trumpet|577|
|Violin|580|
|Human Singing Voice|778|


For the Test Set there are 2874 excerpts in 16 bit stereo wav format sampled at 44.1kHz. 
Considering that these files are not necessarily 3 seconds long, while all of the training files 
are just 3 seconds long, we are using a Sliding Window approach to actually predict the class for 
each test file. The sliding window would be 3 second window slided over 1 second interval. Because 
we are using deep learning architectures to construct our classification model, we have to predefine the 
size of the input vector/matrix and considering that the model will be trained over the Training Set 
containing 3 second audio files, it will only allow inputs of that length, until we use a zero padding framework 
which will not be a included in this project. Because of this inherent dependcy that will only allow input 
lengths to be of 3 seconds, we will use the already mentioned sliding window technique to do inference. 

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