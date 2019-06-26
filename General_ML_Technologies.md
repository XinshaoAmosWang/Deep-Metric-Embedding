# 1 Regularisation/Sample Reweighting

#### Emphasis Regularisation by Gradient Rescaling for Training Deep Neural Networks with Noisy Labels (arXiv 2019)
##### Rethinking data fitting and generalisation: MAE has weak training data fitting ability. Please consider how simple our solution is, which is backed up by our fundamental analysis.
* Paper: https://arxiv.org/pdf/1905.11233.pdf
* Comments, sharing, discussion: https://www.researchgate.net/publication/333418661_Emphasis_Regularisation_by_Gradient_Rescaling_for_Training_Deep_Neural_Networks_with_Noisy_Labels/comments
#### Improving MAE against CCE under Label Noise (arXiv 2019)
##### Rethinking data fitting and generalisation: MAE has weak training data fitting ability. Please consider how simple our solution is, which is backed up by our fundamental analysis.
* Paper: https://arxiv.org/pdf/1903.12141.pdf
* Comments, sharing, discussion: 
https://www.researchgate.net/publication/332070641_Improving_MAE_against_CCE_under_Label_Noise


# 2 Normalisation
## 2.1 Pros and cons of weight normalization vs batch normalization
#### StackExchange: https://stats.stackexchange.com/questions/304755/pros-and-cons-of-weight-normalization-vs-batch-normalization

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


#### Comparison of Batch Normalization and Weight Normalization Algorithms for the Large-scale Image Classification: https://arxiv.org/pdf/1709.08145.pdf

We found that although WN achieves better training accuracy, the final test accuracy is significantly lower (≈6%) than that of BN. This result demonstrates the surprising strength of the BN regularization effect which we were unable to compensate for using standard regularization techniques like dropout and weight decay. We also found that training of deep networks with WN algorithms is significantly less stable compared to BN, limiting their practical applications.

## 2.2 What's the difference between Layer Normalization, Recurrent Batch Normalization (2016), and Batch Normalized RNN (2015)? 
#### StackExchange: https://datascience.stackexchange.com/questions/12956/paper-whats-the-difference-between-layer-normalization-recurrent-batch-normal
* Layer normalization (Ba 2016): Does not use batch statistics. Normalize using the statistics collected from all units within a layer of the current sample. Does not work well with ConvNets.

* Recurrent Batch Normalization (BN) (Cooijmans, 2016; also proposed concurrently by Qianli Liao & Tomaso Poggio, but tested on Recurrent ConvNets, instead of RNN/LSTM): Same as batch normalization. Use different normalization statistics for each time step. You need to store a set of mean and standard deviation for each time step.

* Batch Normalized Recurrent Neural Networks (Laurent, 2015): batch normalization is only applied between the input and hidden state, but not between hidden states. i.e., normalization is not applied over time.

* Streaming Normalization (Liao et al. 2016) : it summarizes existing normalizations and overcomes most issues mentioned above. It works well with ConvNets, recurrent learning and online learning (i.e., small mini-batch or one sample at a time):

* Weight Normalization (Salimans and Kingma 2016): whenever a weight is used, it is divided by its L2 norm first, such that the resulting weight has L2 norm 1. That is, output y=x∗(w/|w|), where x and w denote the input and weight respectively. A scalar scaling factor g is then multiplied to the output y=y∗g. But in my experience g seems not essential for performance (also downstream learnable layers can learn this anyway).

* Cosine Normalization (Luo et al. 2017): weight normalization is very similar to cosine normalization, where the same L2 normalization is applied to both weight and input: y=(x/|x|)∗(w/|w|). Again, manual or automatic differentiation can compute appropriate gradients of x and w.


