


## **<u>Neural Network</u>** 

An artificial neural network is a network of neurons. Neurons in a neural network are arranged in layers. The first and the last layer are called the input and output layers. Input layers have as many neurons as the number of attributes in the data set and the output layer has as many neurons as the number of classes of the target variable (for a classification problem). For a regression problem, the number of neurons in the output layer would be 1 (a numeric variable).



A large neural networks can potentially have extremely complex structures, certain assumptions are made to simplify the way information flows in them:

1. Neurons are arranged in layers and the layers are arranged sequentially.
2. Neurons within the same layer do not interact with each other.
3. All the inputs enter the network through the input layer and all the outputs go out of the network through the output layer.
4. Neurons in consecutive layers are densely connected, i.e. all neurons in layer l are connected to all neurons in layer l+1.
5. Every interconnection in the neural network has a weight associated with it, and every neuron has a bias associated with it.
6. All neurons in a particular layer use the same activation function.



Neural networks are trained on weights and biases.
During training, the neural network learning algorithm fits various models to the training data and selects the best model for prediction. The learning algorithm is trained with a fixed set of hyperparameters - the network structure (number of layers, number of neurons in the input, hidden and output layers etc.). It is trained on the weights and the biases, which are the parameters of the network.

## <u>Feed forward propagation-A simple neural network</u>

![image-20210527162428202](C:\Users\sunny\AppData\Roaming\Typora\typora-user-images\image-20210527162428202.png)

<u>**Figure 1:**</u> A simple neural network with two layers; X represents (Oval shape with circles) input and p represents the probability of each class from output layer(oval with circles) 

In artificial neural networks, the output from one layer is used as input to the next layer. Such networks are called ***feedforward neural networks***. This means there are no loops in the network - information is always fed forward, never fed back.

![image-20210527164426559](C:\Users\sunny\AppData\Roaming\Typora\typora-user-images\image-20210527164426559.png)

**Figure 3 Feed forward propagation:** Input layer (oval with circles) with first layer of neuron (rectangular with circles). We have the input, weight matrix (W),  and the output of the first layer (h) after activation. 



1. W is for weight matrix

2. x stands for input

3. y is the ground truth label

4. p is the probability vector of the predicted output

5. h is the output of the hidden layers (inner layer)

6. superscript stands for layer number

7. subscript stands for the index of the individual neuron

   

![image-20210527164622478](C:\Users\sunny\AppData\Roaming\Typora\typora-user-images\image-20210527164622478.png)

Figure 4: Information flow through a layer 1. 



## Calculation of output (h1) of each neuron happens Feed forward mechanism. 

Feed forward algorithm for a neural network with L hidden layer: 

h^0  = x (0th layer is data/image)

for l in range (0,L)

  	  h^l = activation(W^l , h ^l-1)

   	Pi= 	normalize	(e^(W^0.h^L) )

​      

The above algorithm performs feed forward for a single data point through the neural network

So feed forward is summarized as 



## Backward propagation in Neural Network  

- Training task is to compute the optimal weights by minimizing some cost function.
-  The desired output (output from the last layer) minus the actual output is the cost (or the loss), and we to tune the parameters weights that total cost is minimized.
- Total Loss = L = L1+ L1+ .... (sum of the loses of individual data points ) L1= loss of first data point 
- The loss function is defined as follows:

![image-20210527183224366](C:\Users\sunny\AppData\Roaming\Typora\typora-user-images\image-20210527183224366.png)

The loss function is defined in terms of the network output F(xi) and the ground truth yi. Since F(xi) depends on the weights and biases, the loss, in turn, is a function of (weight,w)	The average loss across all data points is denoted by G(w) which we want to minimize.

For a large neural network, the number of weight elements becomes so large and minimizing the loss with so many parameters is a difficult task. This complex task is achieved using gradient descent. An interesting property of gradient descent which is widely used is the chain rule.

Loss function  Cross entropy is written as 

![image-20210527183650018](C:\Users\sunny\AppData\Roaming\Typora\typora-user-images\image-20210527183650018.png)

**Concept of Back propagation**

![image-20210527183806514](C:\Users\sunny\AppData\Roaming\Typora\typora-user-images\image-20210527183806514.png) 

**Figure 4:** Concept of Back propagation weights are updated in Back propagation. Gradients are calculated in a backward direction starting from dz3. Hence, we'll calculate the gradients in the following sequence:

z= input to neuron after multiplying with weights

h = Output after activation 

p =probability 

w= weights 
