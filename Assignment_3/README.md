#Accuracy for Base Model
0.8119

#Model Definition


newModel = Sequential()

newModel.add(SeparableConv2D(48, 3, 3, activation='relu', input_shape=(32, 32, 3))) #30,3 (output size, receptive field)
newModel.add(BatchNormalization())
newModel.add(Dropout(0.2))

newModel.add(SeparableConv2D(96, kernel_size = (3, 3), activation='relu' )) #28,5
newModel.add(BatchNormalization())
newModel.add(Dropout(0.1))

newModel.add(SeparableConv2D(192, kernel_size = (3, 3),  strides=(2, 2), activation='relu' )) #13,7
newModel.add(BatchNormalization())
newModel.add(Dropout(0.1))

newModel.add(SeparableConv2D(48, 3, 3, activation='relu', )) #11,9
newModel.add(BatchNormalization())
newModel.add(Dropout(0.1))

newModel.add(SeparableConv2D(96, 3, 3, activation='relu', )) #9,11
newModel.add(BatchNormalization())
newModel.add(Dropout(0.1))

newModel.add(SeparableConv2D(192, kernel_size = (3, 3), strides=(2, 2), activation='relu', )) #4, 13
newModel.add(BatchNormalization())
newModel.add(Dropout(0.1))

newModel.add(SeparableConv2D(48, 3, 3, activation='relu', )) #2,17
newModel.add(BatchNormalization())
newModel.add(Dropout(0.1))

newModel.add(SeparableConv2D(num_classes, 1, activation='relu')) # (1, 17)
newModel.add(BatchNormalization())

newModel.add(Flatten())
newModel.add(Dense(num_classes, activation='softmax'))
newModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#50 Epoch Logs

Epoch 1/50
390/390 [==============================] - 54s 138ms/step - loss: 1.8991 - acc: 0.3093 - val_loss: 1.7471 - val_acc: 0.3775
Epoch 2/50
390/390 [==============================] - 41s 105ms/step - loss: 1.4771 - acc: 0.4586 - val_loss: 1.4892 - val_acc: 0.4686
Epoch 3/50
390/390 [==============================] - 41s 105ms/step - loss: 1.2290 - acc: 0.5618 - val_loss: 1.1607 - val_acc: 0.5888
Epoch 4/50
390/390 [==============================] - 41s 105ms/step - loss: 1.0692 - acc: 0.6216 - val_loss: 1.0271 - val_acc: 0.6437
Epoch 5/50
390/390 [==============================] - 41s 106ms/step - loss: 0.9583 - acc: 0.6637 - val_loss: 0.9932 - val_acc: 0.6566
Epoch 6/50
390/390 [==============================] - 41s 105ms/step - loss: 0.8810 - acc: 0.6894 - val_loss: 0.8867 - val_acc: 0.6894
Epoch 7/50
390/390 [==============================] - 41s 106ms/step - loss: 0.8104 - acc: 0.7156 - val_loss: 0.8511 - val_acc: 0.7046
Epoch 8/50
390/390 [==============================] - 41s 106ms/step - loss: 0.7600 - acc: 0.7332 - val_loss: 0.8761 - val_acc: 0.6984
Epoch 9/50
390/390 [==============================] - 41s 106ms/step - loss: 0.7236 - acc: 0.7474 - val_loss: 0.7671 - val_acc: 0.7365
Epoch 10/50
390/390 [==============================] - 41s 105ms/step - loss: 0.6838 - acc: 0.7617 - val_loss: 0.9189 - val_acc: 0.6886
Epoch 11/50
390/390 [==============================] - 41s 105ms/step - loss: 0.6567 - acc: 0.7701 - val_loss: 0.7861 - val_acc: 0.7377
Epoch 12/50
390/390 [==============================] - 41s 106ms/step - loss: 0.6312 - acc: 0.7799 - val_loss: 0.6805 - val_acc: 0.7727
Epoch 13/50
390/390 [==============================] - 41s 105ms/step - loss: 0.6072 - acc: 0.7876 - val_loss: 0.6643 - val_acc: 0.7769
Epoch 14/50
390/390 [==============================] - 41s 106ms/step - loss: 0.5877 - acc: 0.7942 - val_loss: 0.6599 - val_acc: 0.7772
Epoch 15/50
390/390 [==============================] - 41s 106ms/step - loss: 0.5672 - acc: 0.8021 - val_loss: 0.6817 - val_acc: 0.7651
Epoch 16/50
390/390 [==============================] - 41s 106ms/step - loss: 0.5586 - acc: 0.8061 - val_loss: 0.6579 - val_acc: 0.7767
Epoch 17/50
390/390 [==============================] - 41s 105ms/step - loss: 0.5367 - acc: 0.8098 - val_loss: 0.6703 - val_acc: 0.7728
Epoch 18/50
390/390 [==============================] - 41s 106ms/step - loss: 0.5290 - acc: 0.8138 - val_loss: 0.6676 - val_acc: 0.7759
Epoch 19/50
390/390 [==============================] - 41s 105ms/step - loss: 0.5150 - acc: 0.8184 - val_loss: 0.7163 - val_acc: 0.7669
Epoch 20/50
390/390 [==============================] - 41s 105ms/step - loss: 0.4994 - acc: 0.8253 - val_loss: 0.6553 - val_acc: 0.7850
Epoch 21/50
390/390 [==============================] - 41s 105ms/step - loss: 0.4946 - acc: 0.8244 - val_loss: 0.6907 - val_acc: 0.7701
Epoch 22/50
390/390 [==============================] - 41s 105ms/step - loss: 0.4849 - acc: 0.8286 - val_loss: 0.6284 - val_acc: 0.7943
Epoch 23/50
390/390 [==============================] - 41s 105ms/step - loss: 0.4703 - acc: 0.8333 - val_loss: 0.6705 - val_acc: 0.7822
Epoch 24/50
390/390 [==============================] - 41s 105ms/step - loss: 0.4623 - acc: 0.8369 - val_loss: 0.6757 - val_acc: 0.7809
Epoch 25/50
390/390 [==============================] - 41s 105ms/step - loss: 0.4489 - acc: 0.8406 - val_loss: 0.6783 - val_acc: 0.7805
Epoch 26/50
390/390 [==============================] - 41s 105ms/step - loss: 0.4441 - acc: 0.8427 - val_loss: 0.6332 - val_acc: 0.7928
Epoch 27/50
390/390 [==============================] - 41s 105ms/step - loss: 0.4356 - acc: 0.8442 - val_loss: 0.6723 - val_acc: 0.7873
Epoch 28/50
390/390 [==============================] - 41s 105ms/step - loss: 0.4282 - acc: 0.8471 - val_loss: 0.6326 - val_acc: 0.7926
Epoch 29/50
390/390 [==============================] - 41s 105ms/step - loss: 0.4274 - acc: 0.8481 - val_loss: 0.6336 - val_acc: 0.7983
Epoch 30/50
390/390 [==============================] - 41s 105ms/step - loss: 0.4189 - acc: 0.8509 - val_loss: 0.6413 - val_acc: 0.7914
Epoch 31/50
390/390 [==============================] - 41s 106ms/step - loss: 0.4087 - acc: 0.8541 - val_loss: 0.6089 - val_acc: 0.8043
Epoch 32/50
390/390 [==============================] - 41s 106ms/step - loss: 0.4022 - acc: 0.8584 - val_loss: 0.6607 - val_acc: 0.7906
Epoch 33/50
390/390 [==============================] - 41s 105ms/step - loss: 0.3987 - acc: 0.8589 - val_loss: 0.6404 - val_acc: 0.7954
Epoch 34/50
390/390 [==============================] - 41s 105ms/step - loss: 0.3951 - acc: 0.8592 - val_loss: 0.6516 - val_acc: 0.7926
Epoch 35/50
390/390 [==============================] - 41s 105ms/step - loss: 0.3888 - acc: 0.8613 - val_loss: 0.6050 - val_acc: 0.8062
Epoch 36/50
390/390 [==============================] - 41s 105ms/step - loss: 0.3789 - acc: 0.8646 - val_loss: 0.6959 - val_acc: 0.7800
Epoch 37/50
390/390 [==============================] - 41s 105ms/step - loss: 0.3796 - acc: 0.8650 - val_loss: 0.6380 - val_acc: 0.7949
Epoch 38/50
390/390 [==============================] - 41s 105ms/step - loss: 0.3753 - acc: 0.8644 - val_loss: 0.6572 - val_acc: 0.7949
Epoch 39/50
390/390 [==============================] - 41s 105ms/step - loss: 0.3758 - acc: 0.8665 - val_loss: 0.6260 - val_acc: 0.8033
Epoch 40/50
390/390 [==============================] - 41s 105ms/step - loss: 0.3614 - acc: 0.8707 - val_loss: 0.6517 - val_acc: 0.7949
Epoch 41/50
390/390 [==============================] - 41s 105ms/step - loss: 0.3612 - acc: 0.8708 - val_loss: 0.6252 - val_acc: 0.8005
Epoch 42/50
390/390 [==============================] - 41s 105ms/step - loss: 0.3584 - acc: 0.8715 - val_loss: 0.6199 - val_acc: 0.8074
Epoch 43/50
390/390 [==============================] - 41s 105ms/step - loss: 0.3512 - acc: 0.8756 - val_loss: 0.6414 - val_acc: 0.8007
Epoch 44/50
390/390 [==============================] - 41s 105ms/step - loss: 0.3473 - acc: 0.8742 - val_loss: 0.6267 - val_acc: 0.8082
Epoch 45/50
390/390 [==============================] - 41s 104ms/step - loss: 0.3500 - acc: 0.8749 - val_loss: 0.6064 - val_acc: 0.8123
Epoch 46/50
390/390 [==============================] - 41s 105ms/step - loss: 0.3399 - acc: 0.8777 - val_loss: 0.6340 - val_acc: 0.8074
Epoch 47/50
390/390 [==============================] - 41s 105ms/step - loss: 0.3426 - acc: 0.8765 - val_loss: 0.6483 - val_acc: 0.7984
Epoch 48/50
390/390 [==============================] - 41s 105ms/step - loss: 0.3346 - acc: 0.8794 - val_loss: 0.6712 - val_acc: 0.7969
Epoch 49/50
390/390 [==============================] - 41s 105ms/step - loss: 0.3328 - acc: 0.8826 - val_loss: 0.6539 - val_acc: 0.8026
Epoch 50/50
390/390 [==============================] - 41s 105ms/step - loss: 0.3297 - acc: 0.8819 - val_loss: 0.6071 - val_acc: 0.8119
Model took 2067.10 seconds to train
