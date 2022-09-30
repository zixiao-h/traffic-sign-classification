# Traffic sign classification using convolutional neural networks
Part of the CS50's Introduction to Artificial Intelligence with Python course

A neural network to classify images of traffic signs, written using the <TT>tensorflow.keras</TT> framework. Training dataset provided by the German Traffic Sign Recognition Benchmark (GTSRB). A small test dataset is provided in gtsrb-small, while the full data can be downloaded via https://benchmark.ini.rub.de/gtsrb_dataset.html

## Implementation process

Initially, I worked with the gtsrb-small directory to load all the images and get a neural network up and running. I started with a single dense layer of 8 units and ReLU activation which gave around 90% accuracy for the smaller directory, but under 10% for the larger one. Increasing the number of units increased the accuracy but took far too long to train. 

To solve this problem, I found that adding 3x3 convolutional and 2x2 pooling layers drastically improved the speed, effectively by reducing the input size by a factor of 4 each time while preserving information through convolution. Larger pool sizes caused too much loss, so instead I opted to add more convolution-pooling layers of size (2,2). Furthermore, increasing the number of filters learned by the convolution helped, but also took longer, so I experimented with finding a balance. 
These refinements allowed the number of dense layer units and hence the accuracy to be increased. Adding more dense layers took longer, but did not appear to improve accuracy, so I kept a single dense layer.

Ultimately, I chose to use three convolution-pooling layers, with 3x3 kernels learning 16, 32, and 64 filters respectively, with a pool size of 2x2 each time. The processd data was fed into a single dense layer of 1024 units, leading to an output layer of 43 categories with dropout to prevent overfitting. This resulted in an accuracy of around 97%. Changing the activation of the dense layer from ReLU to sigmoid increased this to 99%, but given the small data set I was concerned this was overfit, so I kept ReLU activation.
