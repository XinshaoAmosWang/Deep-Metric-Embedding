#### 1. Sampling Matters in Deep Embedding Learning
Paper: http://openaccess.thecvf.com/content_ICCV_2017/papers/Wu_Sampling_Matters_in_ICCV_2017_paper.pdf

Code (MXNet + Python): https://github.com/apache/incubator-mxnet/tree/master/example/gluon/embedding_learning

Pipeline: net.features->Dense 128->L2 Norm -> Distance Weighted Sampling -> Margin Loss

Automatic Learning Beta : does not help

The margin in Margin Loss is sensitive

