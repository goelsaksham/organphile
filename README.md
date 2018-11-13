# organphile
Repository hosting code for my Introduction to Artificial Intelligence Project - Organphile (An Instrument Recognizer)

## Data Exploration


## Data Creation
This part of the project is responsible to read the data from the given input files and automatically convert the data into format that can be used to train the different machine learning and deep learning algorithms/architectures.


## TODO:
 * Create a Training Data Loader Script
   * Add functions to load training data from the .wav files
   * Zero Padding
   * Categorical variables to Numerical
   * Sliding Window on top of the original loaded data to divide it into the chunks of times 
 * Training Data Augmentation (Model Based)
   * Sliding Window on top of each data sample to convert it to a vector and time based 2-D array model
 * Create a Test Data Loader Script
    * Load the data from the .wav files in the test directory and attach the given class labels for that audio
    
## Results

This section holds the results of the various experiments conducted during the course of the project and record all 
the hyperparameters for the type of training along with the model architecture so that these can be added to the 
final report for the AI project.

    Date    |    Notebook    |    Model    |    Accuracy    |    Precision    |    Recall    |    Epochs    |    Time
     Window    |    Feature Window    |    Notes    
    
    ----|----|----|----|----|----|----|----|----|----
    Date    |    Notebook    |    Model    |    Accuracy    |    Precision    |    Recall    |    Epochs    |    Time
     Window    |    Feature Window    |    Notes    