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

## Data preprocessing
The Jupyter notebook data_preprocess.ipynb contains the code for preprocessing data and creating files for the RNN model.
The input to this notebook is a csv file containing at least 2 columns with names 'sequence' and 'effect'. The 'sequence' column should have sequences (ACGT only) with exactly 23 nucleotides. The file name should be specified inside the notebook.

Other settings that can be modified include prefix of output file name (this will be used in the next step), percent breakdown for training, validation, and testing set, and type of model ('regression' or 'classification').

If you would like to replicate our results on the full dataset:
1. Download the file "GenomeCRISPR_full05112017.csv.gz" from http://www.dkfz.de/signaling/crispr-downloads/GENOMECRISPR/, unzip and place in /data directory
2. Uncomment the corresponding lines in data_preprocess.ipynb:
- data= pd.read_csv('data/GenomeCRISPR_full05112017.csv')
- outfile_prefix = 'PubMed1'   (or outfile_prefix = 'PubMed2')
- pubmedid = 29083409 (or pubmedid = 26780180)
- data = data[data['pubmed'] ==pubmedid]

The output of running this script are 2-6 pickle files storing the data and labels for training, validation, and testing sets (depending on the breakdown).

We have provided sample inputs (data/samplePubMed1.csv) that can be run in this step. With the default settings and this input file, the script should output 6 pickle files (data and labels for training, validation, and testing sets).

## Training the RNN model
Model training can be done in the top section of the CrispRNN.ipynb Jupyter notebook.
The input to this notebook are pickle files created in the previous step. The prefix of the file names must be specified inside the notebook.

Model parameters that can be modified include the type of model ('regression' or 'classification'), learning rate, training steps, display step, number of hidden layers in the LSTM cell, intermediate activation functions in the LSTM cell, and model name to save the trained model as.

The output of running the top section of this script is a trained Tensorflow model with the specified name. As the model is training, the minibatch loss (on 1 sample) is printed out at every display step iteration. Code for plotting a scatter plot to visualize the training loss over time and saving this as a csv file is also included.

Tutorial at this link served as the initial template for model development: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py


## Pre-trained model
We have provided two pre-trained models (one for regression and one for classification) in the models directory. They are named model_r_10m and model_c_10m, respectively. These models were trained using data from Meyers et al. (PMID 29083409) which were retrieved from the full database of CRISPR experiments available for download at [GenomeCRISPR](http://genomecrispr.dkfz.de/). The default setting for selecting a model for prediction is one of the pre-trained models.


## Predicting and visualizing on-target effects
Predictions can be made for both validation and test data sets in the corresponding sections of the CrispRNN.ipynb Jupyter notebook (see headers on where each part are). Note that the first block of code specifying inputs must be run before scrolling down to prediction sections. Either the trained model on the sample subset data provided or the pre-trained model can be used for predictions. The prefix for pickle files for prediction and the model name should be specified inside the notebook.

For both the validation and test set, code is provided to create scatter plots of the predicted effect vs the actual effect, report Spearman correlations, and output csv files for the predictions.


