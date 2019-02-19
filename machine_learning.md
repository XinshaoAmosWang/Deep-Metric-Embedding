# General Machine Learning Technologies
## Normalisation
#### Pros and cons of weight normalization vs batch normalization
* StachExchange: https://stats.stackexchange.com/questions/304755/pros-and-cons-of-weight-normalization-vs-batch-normalization
Batch Norm:
(+) Stable if the batch size is large
(+) Robust (in train) to the scale & shift of input data
(+) Robust to the scale of weight vector
(+) Scale of update decreases while training
(-) Not good for online learning
(-) Not good for RNN, LSTM
(-) Different calculation between train and test

Weight Norm:
(+) Smaller calculation cost on CNN
(+) Well-considered about weight initialization
(+) Implementation is easy
(+) Robust to the scale of weight vector
(-) Compared with the others, might be unstable on training
(-) High dependence to input data

Layer Norm:
(+) Effective to small mini batch RNN
(+) Robust to the scale of input
(+) Robust to the scale and shift of weight matrix
(+) Scale of update decreases while training
(-) Might be not good for CNN (Batch Norm is better in some cases)

* Comparison of Batch Normalization and Weight Normalization Algorithms for the Large-scale Image Classification: https://arxiv.org/pdf/1709.08145.pdf
We found that although WN achieves better training accuracy, the final test accuracy is significantly lower (≈6%) than that of BN. This result demonstrates the surprising strength of the BN regularization effect which we were unable to compensate for using standard regularization techniques like dropout and weight decay. We also found that training of deep networks with WN algorithms is significantly less stable compared to BN, limiting their practical applications.

