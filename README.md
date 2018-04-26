# CE888_ESDA_Assignment2

Evolutionary Strategies for Domain Adaptation

It is often the case that the distribution of the source data is not the same as the target data; for example we only have labeled data examples from images of animals we took in artificial captivity conditions (source data), but we would like to classify animals in the wild (target data). We don't know the labels of the target data, so we have to learn features that fail to discriminate between source and target distributions, but are good enough to actually learn the mapping between those distributions and their labels.

References:
  - Ganin, Yaroslav, and Victor Lempitsky. "Unsupervised domain adaptation by backpropagation." International Conference on Machine Learning. 2015.
  - Saenko, Kate, et al. "Adapting visual category models to new domains." European conference on computer vision. Springer, Berlin, Heidelberg, 2010.
  - InfoGA archive

Data:
  - <a href=https://github.com/pumpikano/tf-dann>MNIST-M Dataset, Blobs and related code</a> (including <a href=https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html#bsds500>BSDS500</a> dataset)
  - <a href=https://github.com/jindongwang/transferlearning/blob/master/doc/dataset.md#office-31>The office 31 Dataset</a>
