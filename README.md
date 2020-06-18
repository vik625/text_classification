# Text Classification

## Importing the dependencies
pip install csv  
pip install pandas  
pip install sci-kit learn  
pip install tensorflow

## Reading and preprocessing of the data
We strip the dataset into two values one of which is "x" which is the input values and the other is "y" which are labels
We encoded the labels using onehotencoder

## Splitting
Splitting of dataset into training set with a ratio of 0.6 and test and validation set into a ratio of 0.4

## Creating a model
We creat a dense layer of 5 units which uses a softmax activation function and  use a loss of crossentropy and  Adams optimizer 

## Training
Fit the Model to the dataset where we train over a batch of 16 and train it for 6 epochs

## Evaluation 
Make the predictions over the test set  
HotEncode the results and print it


## Running the script  
python classify.py
