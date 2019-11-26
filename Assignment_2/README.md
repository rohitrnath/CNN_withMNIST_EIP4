#Logs for 20 Epochs

Train on 60000 samples, validate on 10000 samples
Epoch 1/20

Epoch 00001: LearningRateScheduler setting learning rate to 0.015.
60000/60000 [==============================] - 31s 515us/step - loss: 0.0859 - acc: 0.9738 - val_loss: 0.0597 - val_acc: 0.9808
Epoch 2/20

Epoch 00002: LearningRateScheduler setting learning rate to 0.0113722517.
60000/60000 [==============================] - 11s 190us/step - loss: 0.0570 - acc: 0.9823 - val_loss: 0.0459 - val_acc: 0.9860
Epoch 3/20

Epoch 00003: LearningRateScheduler setting learning rate to 0.0091575092.
60000/60000 [==============================] - 11s 190us/step - loss: 0.0454 - acc: 0.9857 - val_loss: 0.0279 - val_acc: 0.9914
Epoch 4/20

Epoch 00004: LearningRateScheduler setting learning rate to 0.0076647931.
60000/60000 [==============================] - 12s 192us/step - loss: 0.0401 - acc: 0.9872 - val_loss: 0.0337 - val_acc: 0.9898
Epoch 5/20

Epoch 00005: LearningRateScheduler setting learning rate to 0.0065905097.
60000/60000 [==============================] - 11s 190us/step - loss: 0.0355 - acc: 0.9888 - val_loss: 0.0315 - val_acc: 0.9915
Epoch 6/20

Epoch 00006: LearningRateScheduler setting learning rate to 0.0057803468.
60000/60000 [==============================] - 11s 189us/step - loss: 0.0322 - acc: 0.9892 - val_loss: 0.0214 - val_acc: 0.9939
Epoch 7/20

Epoch 00007: LearningRateScheduler setting learning rate to 0.0051475635.
60000/60000 [==============================] - 11s 190us/step - loss: 0.0291 - acc: 0.9903 - val_loss: 0.0240 - val_acc: 0.9919
Epoch 8/20

Epoch 00008: LearningRateScheduler setting learning rate to 0.0046396536.
60000/60000 [==============================] - 11s 189us/step - loss: 0.0281 - acc: 0.9909 - val_loss: 0.0226 - val_acc: 0.9930
Epoch 9/20

Epoch 00009: LearningRateScheduler setting learning rate to 0.004222973.
60000/60000 [==============================] - 11s 189us/step - loss: 0.0272 - acc: 0.9912 - val_loss: 0.0252 - val_acc: 0.9922
Epoch 10/20

Epoch 00010: LearningRateScheduler setting learning rate to 0.0038749677.
60000/60000 [==============================] - 11s 188us/step - loss: 0.0253 - acc: 0.9919 - val_loss: 0.0300 - val_acc: 0.9916
Epoch 11/20

Epoch 00011: LearningRateScheduler setting learning rate to 0.0035799523.
60000/60000 [==============================] - 11s 190us/step - loss: 0.0240 - acc: 0.9926 - val_loss: 0.0284 - val_acc: 0.9927
Epoch 12/20

Epoch 00012: LearningRateScheduler setting learning rate to 0.00332668.
60000/60000 [==============================] - 11s 190us/step - loss: 0.0219 - acc: 0.9927 - val_loss: 0.0266 - val_acc: 0.9935
Epoch 13/20

Epoch 00013: LearningRateScheduler setting learning rate to 0.0031068766.
60000/60000 [==============================] - 11s 189us/step - loss: 0.0220 - acc: 0.9929 - val_loss: 0.0287 - val_acc: 0.9924
Epoch 14/20

Epoch 00014: LearningRateScheduler setting learning rate to 0.002914319.
60000/60000 [==============================] - 11s 188us/step - loss: 0.0208 - acc: 0.9937 - val_loss: 0.0229 - val_acc: 0.9930
Epoch 15/20

Epoch 00015: LearningRateScheduler setting learning rate to 0.0027442371.
60000/60000 [==============================] - 11s 191us/step - loss: 0.0199 - acc: 0.9937 - val_loss: 0.0230 - val_acc: 0.9935
Epoch 16/20

Epoch 00016: LearningRateScheduler setting learning rate to 0.0025929127.
60000/60000 [==============================] - 11s 191us/step - loss: 0.0180 - acc: 0.9942 - val_loss: 0.0234 - val_acc: 0.9938
Epoch 17/20

Epoch 00017: LearningRateScheduler setting learning rate to 0.002457405.
60000/60000 [==============================] - 11s 190us/step - loss: 0.0188 - acc: 0.9941 - val_loss: 0.0220 - val_acc: 0.9940
Epoch 18/20

Epoch 00018: LearningRateScheduler setting learning rate to 0.0023353573.
60000/60000 [==============================] - 11s 190us/step - loss: 0.0182 - acc: 0.9942 - val_loss: 0.0202 - val_acc: 0.9946
Epoch 19/20

Epoch 00019: LearningRateScheduler setting learning rate to 0.0022248591.
60000/60000 [==============================] - 11s 189us/step - loss: 0.0183 - acc: 0.9937 - val_loss: 0.0227 - val_acc: 0.9939
Epoch 20/20

Epoch 00020: LearningRateScheduler setting learning rate to 0.002124345.
60000/60000 [==============================] - 11s 189us/step - loss: 0.0168 - acc: 0.9945 - val_loss: 0.0205 - val_acc: 0.9946
<keras.callbacks.History at 0x7f52e1fd7160>

#model.evaluate
[0.020545518947319943, 0.9946]

#Strategy

1. First of all, re-arranged the model with the order of convolution block first then transition block manner.
2. Tried with different <dropout rate>, finally fixed with Dropout(0.15)
3. Tried different <learning rates>. Its observed that, values greater than the given values gives better result. So choose learning rate = 0.015 randomly.
4. Tried with different <batch size>. Seems like higher batch size giving faster execution. Finally used batch size =128
5. Instead of Flatten() tried with GlobalAveragePooling2D(). Flatten() seems to be kind of fully connected layer, that is why tried with GAP.
