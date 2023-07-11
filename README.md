# CIFAR-10-Custom-Neural-Network-Project
A solution to the CIFAR-10 image classification problem in PyTorch, implementing a custom architecture following the mentioned specifications. Final validation accuracy of about 87.9%.
The Goals :
• To implement a specific model to solve the CIFAR-10 classification problem and classify every image in terms of 1 out of 10 classes.

• To implement the training pipeline to train the model and achieve the highest accuracy possible.

The Model :
An architecture to process images based on Convolutional Neural Networks, consists of Backbone (B1,...,Bn) and a Classifier.

The Backbone :
The backbone consists of 11 blocks. Each block is an instance of the Block class in the code. The implementation for each block includes:

• A Linear/MLP layer predicting a vector a = [a1, a2, ..., ak] with K elements from the input tensor X. In our code, this is achieved by applying the fully connected layer self.fc on the output of the adaptive average pooling self.avg_pool(x).view(b, c) (where x is the input tensor).

• K Conv layers are combined using vector a to produce a single output:

O = a1 * Conv1(x) + ... + ak * Conv_k(x)

This is accomplished in the forward method of the Block class with the following loop:

for i, conv in enumerate(self.convs): out += a[:, i:i + 1] * conv(x)

The Classifier :
The classifier takes as input the output of the last block (B11).

It computes a mean feature f = SpatialAveragePool(O_11), where O_11 is the output of the 11th block. In our code, this is achieved by the nn.AdaptiveAvgPool2d((1, 1)) layer in the self.classifier module.

The mean feature f is then passed to a classifier, which, in our case, is a fully connected layer with a softmax activation (nn.Linear(512, 10)). This layer produces the final class scores for the input image.
