
# Model Graph
        ----------------------------------------------------------------
                Layer (type)               Output Shape         Param #
        ================================================================
                    Conv2d-1           [-1, 12, 26, 26]             120
                      ReLU-2           [-1, 12, 26, 26]               0
               BatchNorm2d-3           [-1, 12, 26, 26]              24
                 Dropout2d-4           [-1, 12, 26, 26]               0
                    Conv2d-5           [-1, 12, 24, 24]           1,308
                      ReLU-6           [-1, 12, 24, 24]               0
               BatchNorm2d-7           [-1, 12, 24, 24]              24
                 Dropout2d-8           [-1, 12, 24, 24]               0
                    Conv2d-9           [-1, 24, 22, 22]           2,616
                     ReLU-10           [-1, 24, 22, 22]               0
              BatchNorm2d-11           [-1, 24, 22, 22]              48
                Dropout2d-12           [-1, 24, 22, 22]               0
                   Conv2d-13           [-1, 12, 22, 22]             300
                     ReLU-14           [-1, 12, 22, 22]               0
               BatchNorm2d-15           [-1, 12, 22, 22]              24
                AvgPool2d-16           [-1, 12, 11, 11]               0
                   Conv2d-17             [-1, 16, 9, 9]           1,744
                     ReLU-18             [-1, 16, 9, 9]               0
              BatchNorm2d-19             [-1, 16, 9, 9]              32
                Dropout2d-20             [-1, 16, 9, 9]               0
                   Conv2d-21             [-1, 32, 7, 7]           4,640
                     ReLU-22             [-1, 32, 7, 7]               0
               BatchNorm2d-23             [-1, 32, 7, 7]              64
                Dropout2d-24             [-1, 32, 7, 7]               0
                   Conv2d-25             [-1, 16, 7, 7]             528
                     ReLU-26             [-1, 16, 7, 7]               0
              BatchNorm2d-27             [-1, 16, 7, 7]              32
                AvgPool2d-28             [-1, 16, 3, 3]               0
                   Linear-29                   [-1, 10]           1,450
        ================================================================
        Total params: 12,954
        Trainable params: 12,954
        Non-trainable params: 0
        ----------------------------------------------------------------
        
# Logs


  0%|          | 0/469 [00:00<?, ?it/s]

Epoch 1

loss=0.05895450338721275 batch_id=468: 100%|██████████| 469/469 [00:12<00:00, 38.44it/s]

Train set: Average loss: 0.0020, Accuracy: 55632/60000 (92.72%)

  0%|          | 0/469 [00:00<?, ?it/s]

Test set: Average loss: 0.0494, Accuracy: 9844/10000 (98.44%)

Epoch 2

loss=0.19589729607105255 batch_id=468: 100%|██████████| 469/469 [00:12<00:00, 37.76it/s]

Train set: Average loss: 0.0006, Accuracy: 58634/60000 (97.72%)

  0%|          | 0/469 [00:00<?, ?it/s]

Test set: Average loss: 0.0362, Accuracy: 9884/10000 (98.84%)

Epoch 3

loss=0.05269436538219452 batch_id=468: 100%|██████████| 469/469 [00:12<00:00, 37.42it/s]

Train set: Average loss: 0.0004, Accuracy: 58955/60000 (98.26%)

  0%|          | 0/469 [00:00<?, ?it/s]

Test set: Average loss: 0.0300, Accuracy: 9894/10000 (98.94%)

Epoch 4

loss=0.05412048473954201 batch_id=468: 100%|██████████| 469/469 [00:12<00:00, 37.38it/s]

Train set: Average loss: 0.0004, Accuracy: 59109/60000 (98.52%)

  0%|          | 0/469 [00:00<?, ?it/s]

Test set: Average loss: 0.0287, Accuracy: 9913/10000 (99.13%)

Epoch 5

loss=0.017384842038154602 batch_id=468: 100%|██████████| 469/469 [00:12<00:00, 37.55it/s]

Train set: Average loss: 0.0003, Accuracy: 59204/60000 (98.67%)

  0%|          | 0/469 [00:00<?, ?it/s]

Test set: Average loss: 0.0240, Accuracy: 9924/10000 (99.24%)

Epoch 6

loss=0.020023087039589882 batch_id=468: 100%|██████████| 469/469 [00:12<00:00, 37.37it/s]

Train set: Average loss: 0.0003, Accuracy: 59249/60000 (98.75%)

  0%|          | 0/469 [00:00<?, ?it/s]

Test set: Average loss: 0.0219, Accuracy: 9931/10000 (99.31%)

Epoch 7

loss=0.06728097051382065 batch_id=468: 100%|██████████| 469/469 [00:12<00:00, 37.54it/s]

Train set: Average loss: 0.0003, Accuracy: 59252/60000 (98.75%)

  0%|          | 0/469 [00:00<?, ?it/s]

Test set: Average loss: 0.0227, Accuracy: 9923/10000 (99.23%)

Epoch 8

loss=0.003209652379155159 batch_id=468: 100%|██████████| 469/469 [00:12<00:00, 37.84it/s]

Train set: Average loss: 0.0003, Accuracy: 59322/60000 (98.87%)

  0%|          | 0/469 [00:00<?, ?it/s]

Test set: Average loss: 0.0205, Accuracy: 9942/10000 (99.42%)

Epoch 9

loss=0.030697999522089958 batch_id=468: 100%|██████████| 469/469 [00:12<00:00, 37.24it/s]

Train set: Average loss: 0.0003, Accuracy: 59380/60000 (98.97%)

  0%|          | 0/469 [00:00<?, ?it/s]

Test set: Average loss: 0.0219, Accuracy: 9937/10000 (99.37%)

Epoch 10

loss=0.014375305734574795 batch_id=468: 100%|██████████| 469/469 [00:12<00:00, 37.42it/s]

Train set: Average loss: 0.0002, Accuracy: 59474/60000 (99.12%)

  0%|          | 0/469 [00:00<?, ?it/s]

Test set: Average loss: 0.0190, Accuracy: 9942/10000 (99.42%)

Epoch 11

loss=0.01940385065972805 batch_id=468: 100%|██████████| 469/469 [00:12<00:00, 38.29it/s]

Train set: Average loss: 0.0002, Accuracy: 59506/60000 (99.18%)

  0%|          | 0/469 [00:00<?, ?it/s]

Test set: Average loss: 0.0190, Accuracy: 9942/10000 (99.42%)

Epoch 12

loss=0.023499831557273865 batch_id=468: 100%|██████████| 469/469 [00:12<00:00, 38.02it/s]

Train set: Average loss: 0.0002, Accuracy: 59506/60000 (99.18%)

  0%|          | 0/469 [00:00<?, ?it/s]

Test set: Average loss: 0.0184, Accuracy: 9945/10000 (99.45%)

Epoch 13

loss=0.034169454127550125 batch_id=468: 100%|██████████| 469/469 [00:12<00:00, 37.59it/s]

Train set: Average loss: 0.0002, Accuracy: 59509/60000 (99.18%)

  0%|          | 0/469 [00:00<?, ?it/s]

Test set: Average loss: 0.0185, Accuracy: 9943/10000 (99.43%)

Epoch 14

loss=0.01752757467329502 batch_id=468: 100%|██████████| 469/469 [00:12<00:00, 37.47it/s]

Train set: Average loss: 0.0002, Accuracy: 59502/60000 (99.17%)

  0%|          | 0/469 [00:00<?, ?it/s]

Test set: Average loss: 0.0183, Accuracy: 9940/10000 (99.40%)

Epoch 15

loss=0.015287205576896667 batch_id=468: 100%|██████████| 469/469 [00:12<00:00, 37.63it/s]

Train set: Average loss: 0.0002, Accuracy: 59519/60000 (99.20%)

  0%|          | 0/469 [00:00<?, ?it/s]

Test set: Average loss: 0.0185, Accuracy: 9941/10000 (99.41%)

Epoch 16

loss=0.022106455639004707 batch_id=468: 100%|██████████| 469/469 [00:12<00:00, 36.96it/s]

Train set: Average loss: 0.0002, Accuracy: 59506/60000 (99.18%)

  0%|          | 0/469 [00:00<?, ?it/s]

Test set: Average loss: 0.0181, Accuracy: 9943/10000 (99.43%)

Epoch 17

loss=0.014538110233843327 batch_id=468: 100%|██████████| 469/469 [00:12<00:00, 37.62it/s]

Train set: Average loss: 0.0002, Accuracy: 59522/60000 (99.20%)

  0%|          | 0/469 [00:00<?, ?it/s]

Test set: Average loss: 0.0180, Accuracy: 9945/10000 (99.45%)

Epoch 18

loss=0.016605885699391365 batch_id=468: 100%|██████████| 469/469 [00:12<00:00, 37.20it/s]

Train set: Average loss: 0.0002, Accuracy: 59512/60000 (99.19%)

  0%|          | 0/469 [00:00<?, ?it/s]

Test set: Average loss: 0.0179, Accuracy: 9946/10000 (99.46%)

Epoch 19

loss=0.027702972292900085 batch_id=468: 100%|██████████| 469/469 [00:12<00:00, 37.38it/s]

Train set: Average loss: 0.0002, Accuracy: 59534/60000 (99.22%)

Test set: Average loss: 0.0183, Accuracy: 9944/10000 (99.44%)


