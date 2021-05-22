# Assignment 3 
### Write a neural network that can:

1) take 2 inputs:
    1) an image from MNIST dataset, and
    2) a random number between 0 and 9
2) and gives two outputs:
    1) the "number" that was represented by the MNIST image, and
    2) the "sum" of this number with the random number that was generated and sent as the input to the network
    ![network](https://cdn.inst-fs-iad-prod.inscloudgate.net/1af0cd6a-b92b-4c38-abad-77a5d54129c7/assign.png?token=eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCIsImtpZCI6ImNkbiJ9.eyJyZXNvdXJjZSI6Ii8xYWYwY2Q2YS1iOTJiLTRjMzgtYWJhZC03N2E1ZDU0MTI5YzcvYXNzaWduLnBuZyIsInRlbmFudCI6ImNhbnZhcyIsInVzZXJfaWQiOiI3MDAwMDAyMzcyNDEzNCIsImlhdCI6MTYyMTY2NjgwMiwiZXhwIjoxNjIxNzUzMjAyfQ.xvUFwPHaT6d-0nVrRBLP1XhaiZrByCz_bI3fjmgfzhN_kzyO5zky34uRTIEbC4LSYHn6SLMfVIK6PjDBT-o8EA&download=1&content_type=image%2Fpng)

## Implementation
### 1) Data preparation
    a) To prepare random number dataset torch.randint is used as it gives similar count of each number
    b) 2 lists are of size of mnist trainset and testset are created 
    c) In custom dataloader same index form minist and random list are returned
    d) getitem returns (mnist_image, mnist_label, random_number_one_hot_encoded, sum_of_mnist_label_and_random_number)

### 2) Model
#### a) Model Graph
Net(
  * (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  * (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  * (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  * (pintconv1): Conv2d(64, 16, kernel_size=(1, 1), stride=(1, 1))
  * (conv3): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  * (conv4): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  * (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  * (pintconv2): Conv2d(64, 16, kernel_size=(1, 1), stride=(1, 1))
  * (conv5): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1))
  * (conv6): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
  * (conv7): Conv2d(64, 10, kernel_size=(3, 3), stride=(1, 1))
  * (fc1): Linear(in_features=20, out_features=128, bias=True)
  * (fc2): Linear(in_features=128, out_features=18, bias=True)
)

#### b) inputs (mnist and random number one hot encoded)
1) mnist image passes through convolution layers 
2) last convolutuion later is concatenated with one hot encoded random number and passes through dense layers

#### c) outputs
1) log_softmax of last convlayer is output of mnist prediction
2) log_softmax of last dense layer is output of sum prediction
    
#### d) loss   
1) Crossentropy loss is used for both minst and sum as the result is in range of 0 and 9 for mnist and 0 and 18 for sum
2) total_loss is average of both losses

#### e) logs and evaluation

Model reached 99% accuracy for test dataset for both mnist and sum in 3 epochs

Epoch 1
/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:37: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
loss=0.6682172417640686 batch_id=1874: 100%|██████████| 1875/1875 [00:27<00:00, 67.77it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 7155.6940, Mnist_Accuracy: 9678/10000 (97%), Sum_Accuracy: 6973/10000 (70%)

Epoch 2
loss=0.057370174676179886 batch_id=1874: 100%|██████████| 1875/1875 [00:26<00:00, 69.84it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 1180.7436, Mnist_Accuracy: 9841/10000 (98%), Sum_Accuracy: 9792/10000 (98%)

Epoch 3
loss=0.017671098932623863 batch_id=1874: 100%|██████████| 1875/1875 [00:26<00:00, 70.67it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 577.6102, Mnist_Accuracy: 9874/10000 (99%), Sum_Accuracy: 9870/10000 (99%)

Epoch 4
loss=0.014177262783050537 batch_id=1874: 100%|██████████| 1875/1875 [00:26<00:00, 69.50it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 393.3417, Mnist_Accuracy: 9933/10000 (99%), Sum_Accuracy: 9902/10000 (99%)

Epoch 5
loss=0.016728326678276062 batch_id=1874: 100%|██████████| 1875/1875 [00:26<00:00, 69.75it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 380.4174, Mnist_Accuracy: 9917/10000 (99%), Sum_Accuracy: 9886/10000 (99%)

Epoch 6
loss=0.0042080688290297985 batch_id=1874: 100%|██████████| 1875/1875 [00:26<00:00, 70.44it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 330.7261, Mnist_Accuracy: 9923/10000 (99%), Sum_Accuracy: 9905/10000 (99%)

Epoch 7
loss=0.034932319074869156 batch_id=1874: 100%|██████████| 1875/1875 [00:26<00:00, 69.66it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 338.5737, Mnist_Accuracy: 9922/10000 (99%), Sum_Accuracy: 9902/10000 (99%)

Epoch 8
loss=0.0793527215719223 batch_id=1874: 100%|██████████| 1875/1875 [00:26<00:00, 70.19it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 282.7241, Mnist_Accuracy: 9929/10000 (99%), Sum_Accuracy: 9925/10000 (99%)

Epoch 9
loss=0.007101314142346382 batch_id=1874: 100%|██████████| 1875/1875 [00:26<00:00, 71.21it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 430.1920, Mnist_Accuracy: 9902/10000 (99%), Sum_Accuracy: 9894/10000 (99%)

Epoch 10
loss=0.00295666023157537 batch_id=1874: 100%|██████████| 1875/1875 [00:26<00:00, 70.96it/s]
Test set: Average loss: 279.5039, Mnist_Accuracy: 9932/10000 (99%), Sum_Accuracy: 9920/10000 (99%)
