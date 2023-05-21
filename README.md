# DL_Assignment3
## Author : Adarsh Gupta CS22M006
## Overview
The project is to 
* model sequence-to-sequence learning problems using Recurrent Neural Networks 
* compare different cells such as vanilla RNN, LSTM and GRU 
* understand how attention networks overcome the limitations of vanilla seq2seq models

## Folder structure
* **withoutAttention.ipynb:** Contains code to run sweep with various hyperparameter configurations. This also contain implementation of various cells like RNN, LSTM and GRU. It also contains all the functions which are required to train the modelwithout attention.
* **trainWithoutAttention.py:** It contains code to train the seq to seq model without attention, as well as code to support the training of model using command line interface. Used **ArgParse** to support this feature.
* **withAttention.ipynb:** Contains code to run sweep with various hyperparameter configurations. This also contain implementation of various cells like RNN, LSTM and GRU. It also contains all the functions which are required to train the model with attention.
* **trainWithAttention.py:** It contains code to train the seq to seq model with attention, as well as code to support the training of model using command line interface. Used **ArgParse** to support this feature.
* **wronglyPredictedWordsWithoutAttention.txt:** It contains all the wrongly predicted words and its actual transliteration on validation dataset on the best model without attention
* **wronglyPredictedWordsWithAttention.txt:** It contains all the wrongly predicted words and its actual transliteration on validation dataset on the best model with attention
* **testDatasetWordsWithoutAttention.txt:** It contains all the words (predicted and its actual transliteration on test dataset on the best model without attention
* **testDatasetWordsWithAttention.txt:** It contains all the words (predicted and its actual transliteration on test dataset on the best model with attention
 
## Instructions to train and evaluate various models

1. Run the trainWithoutAttention.py using command line to model without attention and pass parameters which you want to set. Passing parameters is optional, if you don't want to pass parameters then it will take default hyperparameter values to train the model.
Here is one example of the command to run trainWithoutAttention.py and train the model.

`
python trainWithoutAttention.py -wp "Assignment 3" -we "cs22m006" -es 128 -bs 256 --cell_type "LSTM" --epochs 20
`

2. Run the trainWithAttention.py using command line to model with attention and pass parameters which you want to set. Passing parameters is optional, if you don't want to pass parameters then it will take default hyperparameter values to train the model.
Here is one example of the command to run trainWithAttention.py and train the model.

`
python trainWithAttention.py -wp "Assignment 3" -we "cs22m006" -es 128 -bs 256 --cell_type "LSTM" --epochs 20
`

3. After running the command, it will print accuracies and loss. It will also log accuracies and loss in wandb.

4. **Pass wandb entity as your wandb username only, otherwise it will give error.**

# Report Link
