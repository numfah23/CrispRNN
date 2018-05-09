# CrispRNN
Computational Genomics (COMS 4761) Project: CrispRNN: Predicting CRISPR sgRNA effect

CrisprRNN is a bidirectional Recurrent Neural Net-work (RNN) with long short-term memory units (LSTM) built user Tensorflow to predict on-target effect of sgRNAs in CRISPR-Cas9 using sequence information.


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

The other libraries can also be installed using pip:
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
The code is contained in the following 2 files: final_data_preprocess.ipynb and final_biRNN.ipynb.


# Test-run on sample inputs

## Data preprocessing
The Jupyter notebook final_data_preprocess.ipynb contains the code for preprocessing data and creating files for the RNN model.
The input to this notebook is a csv file containing at least 2 columns with names 'sequence' and 'effect'. The 'sequence' column should have sequences (ACGT only) with exactly 23 nucleotides. The file name should be specified inside the notebook.

Other settings that can be modified include prefix of output file name (this will be used in the next step), percent breakdown for training, validation, and testing set, and type of model ('regression' or 'classification').

The output of running this script are 2-6 pickle files storing the data and labels for training, validation, and testing sets (depending on the breakdown).

We have provided sample inputs (data/samplePubMed1.csv) that can be run in this step. With the default settings and this input file, the script should output 6 pickle files (data and labels for training, validation, and testing sets).

## Training the RNN model
Model training can be done in the top section of the final_biRNN.ipynb Jupyter notebook.
The input to this notebook are pickle files created in the previous step. The prefix of the file names must be specified inside the notebook.

Model parameters that can be modified include the type of model ('regression' or 'classification'), learning rate, training steps, display step, number of hidden layers in the LSTM cell, intermediate activation functions in the LSTM cell, and model name (TODO: add this to top block in code).

The output of running the top section of this script is a trained Tensorflow model with the specified name. As the model is training, the minibatch loss (on 1 sample) is printed out at every display step iteration. Code for plotting a scatter plot to visualize the training loss over time can be found at the bottom at the notebook.

## Pre-trained model
We have provided two pre-trained models (one for regression and one for classification) in the models directory. They are named model_r_10m and model_c_10m, respectively. (TODO: maybe change these names to something simpler?). These models were trained using data from Meyers et al. (PMID 29083409) which were retrieved from the full database of CRISPR experiments available for download at [GenomeCRISPR](http://genomecrispr.dkfz.de/)

## Predicting and visualizing on-target effects
Predictions can be made for both validation and test data sets in the bottom section of the final_biRNN.ipynb Jupyter notebook. (TODO: clearly split between training + predicting sections OR create another notebook for predicting?) Either the trained model on the sample subset data provided or the pre-trained model can be used for predictions. The prefix for pickle files for prediction and the model name should be specified inside the notebook.

For both the validation and test set, this portion of the notebook will create scatter plots of the predicted effect vs the actual effect, report Spearman correlations, and output csv files for the predictions.


