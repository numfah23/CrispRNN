# CrispRNN
Computational Genomics (COMS 4761) Project: CrispRNN: Predicting CRISPR sgRNA effect

CrisprRNN is a bidirectional Recurrent Neural Network (RNN) with long short-term memory units (LSTM) built using Tensorflow to predict on-target effect of sgRNAs in CRISPR-Cas9 using sequence information.


## Prerequisites
Our model depends on the following to be installed: [Python 2](https://www.python.org/downloads/release/python-2715/), Tensorflow, Jupyter notebook, Numpy, Pandas, Pickle, MatplotLib, and SciPy

To install Tensorflow, use one of the following commands:
``` 
# For CPU
pip install tensorflow
# For GPU
pip install tensorflow-gpu
```
For more details about TensorFlow installation, see the [Tensorflow Installation Guide](https://www.tensorflow.org/install/)

The other dependencies can also be installed using pip:
```
pip install jupyter
pip install numpy
pip install pandas
pip install matplotlib
pip install scipy
```
(Note that pickle does not require installation for Python 2)

## Installation
To download and run our code, clone this repository:
```
git clone https://github.com/numfah23/CrispRNN
```
The code is contained in the following 2 Jupyter notebooks: CrispRNN.ipynb and data_preprocess.ipynb


# Test-run on sample inputs

## Data preprocessing for labelled data
The top section of data_preprocess.ipynb contains the code for preprocessing labelled data and creating files for the RNN model.

The input to this section is a csv file containing at least 2 columns with names 'sequence' and 'effect'. (If 'effect' is not available, skip this part and look at the Data preprocessing for unlabelled data section below). The 'sequence' column should have sequences (ACGT only) with exactly 23 nucleotides. The file name should be specified inside the notebook.

Other settings that can be modified include prefix of output file name (this will be used in the next step), percent breakdown for training, validation, and testing set, and type of model ('regression' or 'classification').

The output of running this script is 2-6 pickle files storing the data and labels for training, validation, and testing sets (depending on the breakdown).

We have provided a sample labelled input file (data/samplePubMed1.csv) that can be run in this step. With the default settings and this input file, the script will create 6 pickle files (data and labels for training, validation, and testing sets) that can be used for training or testing the model.

An additional sample labelled input file (data/samplePubMed2.csv) is provided to simulate an independent test set. In conjunction with the uncommented line "n_train, n_val, n_test = 0, 0, 1", the script will generate a test set of pickle files from this input file (or another input file, if desired).

In order to replicate our results on the full dataset:
1. Download the file "GenomeCRISPR_full05112017.csv.gz" from http://www.dkfz.de/signaling/crispr-downloads/GENOMECRISPR/, unzip and place in /data directory
2. Uncomment the corresponding lines in data_preprocess.ipynb:
```
data= pd.read_csv('data/GenomeCRISPR_full05112017.csv')
outfile_prefix = 'PubMed1'   (or outfile_prefix = 'PubMed2')
pubmedid = 29083409 (or pubmedid = 26780180)
data = data[data['pubmed'] ==pubmedid]
```

## Data preprocessing for unlabelled data
The bottom section of data_preprocess.ipynb contains the code for preprocessing unlabelled data and creating files for the RNN model.

The input to this section is a csv file containing at least 1 column called 'sequence' that has sequences (ACGT only) with exactly 23 nucleotides. The file name should be specified inside the notebook under this section.

The output of running this script is a single pickle file storing the one-hot encoded data made for testing predictions only.

We have provided a sample unlabelled input file (data/sample_unlabelled.csv) that can be run in this step.

## Training the RNN model
Model training can be done in the top section of the CrispRNN.ipynb Jupyter notebook.
The input to this notebook is pickle files created in the previous step. The prefix of the file names must be specified inside the notebook.

Model parameters that can be modified include the type of model ('regression' or 'classification'), learning rate, training steps, display step, number of hidden layers in the LSTM cell, intermediate activation functions in the LSTM cell, and model name to save the trained model as.

The output of running the top section of this script is a trained Tensorflow model with the specified name. As the model is training, the minibatch loss (on 1 sample) is printed out at every display step iteration. Code for plotting a scatter plot to visualize the training loss over time and saving this as a csv file is also included.

Tutorial at this link served as the initial template for model development: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py


## Pre-trained model
We have provided two pre-trained models (one for regression and one for classification) in the models directory. They are named model_r_10m and model_c_10m, respectively. These models were trained using data from Meyers et al. (PMID 29083409) which were retrieved from the full database of CRISPR experiments available for download at [GenomeCRISPR](http://genomecrispr.dkfz.de/). The default setting for selecting a model for prediction is one of the pre-trained models.


## Predicting and visualizing on-target effects for labelled data
Predictions can be made for both validation and test data sets in the corresponding sections of the CrispRNN.ipynb Jupyter notebook (see headers indicating each part). Note that the blocks of code preceding the Training section specifying inputs must be run before scrolling down to prediction sections. Either the trained model on the sample subset data provided or the pre-trained model can be used for predictions (the default setting is the pretrained model). The prefix for pickle files for prediction and the model name should be specified inside the notebook. Make sure that the type of the model and actual model are compatible with the labels of the data (ex. model_name_pred = model_r_10m and model_type = 'regression' for preprocessed data created with 'regression' type).

For both the validation and test set, code is provided to create scatter plots of the predicted effect vs the actual effect, report Spearman correlations, and output csv files for the predictions.

## Predicting on-target effects for unlabelled data
Predictions can be made for unlabelled test data sets using any trained model at the bottom of the CrispRNN.ipynb Jupyter notebook (see header for Prediction for unlabelled test set). Note that once again the blocks of code preceding the Training section specifying inputs must be run before scrolling down to the prediction section. In order to be able to make predictions for the correct file, make sure that the test_name_unlab variable is set to the prefix of the pickle file created in the data preprocessing step. Again, make sure that the specified type of the model is the same as the actual model type (ex. model_name_pred = model_r_10m works with model_type = 'regression' and model_c_10m works with classification).

The code in this section will not report any accuracy or loss (since there is no true label) or create any plots. It will simply output a csv file with predicted on-target effects for these sequences.