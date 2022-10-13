

from keras.models import Sequential,  Model
from keras.layers import Dense, Dropout, Input, concatenate

import numpy


train = numpy.loadtxt("train.csv", delimiter=",")
s = train.shape[1] - 1
print(s)
cl = int(numpy.amax(train[:, s]) + 1)
print(f'Количество классов: {cl}')

X1 = train[:, 0:s]

Y = train[:, s]
print(Y)

model_in1 = Input(shape=(s,))
model_out11 = Dense(s*30, input_dim=s, activation="relu", name="layer1")(model_in1)
model_out111 = Dense(s*30, activation="swish", name="layer111")(model_out11)
model_out1 = Dense(s*15, activation="relu", name="layer11")(model_out111)
model1 = Model(model_in1,model_out1)

model_in2 = Input(shape=(s,))
model_out22 = Dense(s*15, input_dim=s, activation="relu", name="layer2")(model_in2)
model_out2 = Dense(s*8, activation="swish", name="layer22")(model_out22)
model2 = Model(model_in2,model_out2)

model_in3 = Input(shape=(s,))
model_out3 = Dense(s*8, input_dim=s, activation="swish", name="layer3")(model_in3)
model3 = Model(model_in3,model_out3)

model_in4 = Input(shape=(s,))
model_out4 = Dense(s*4, input_dim=s, activation="swish", name="layer4")(model_in4)
model4 = Model(model_in4,model_out4)

concatenated = concatenate([model_out1,model_out2,model_out3,model_out4])
out = Dense(cl, activation="softmax", name="outputlayer")(concatenated)

merged_model = Model([model_in1,model_in2, model_in3,model_in4], out)
merged_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=["accuracy"])
merged_model.fit([X1,X1,X1,X1], Y, batch_size=5, epochs=10, verbose=1, validation_split=0.15)
merged_model.save(f"adress")
