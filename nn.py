from packages import *
from params import *

def activation_final(tensorOut):
    out_e = tf.keras.activations.linear(tensorOut[...,equity])
    out_b = tf.keras.activations.linear(tensorOut[...,bond])
    out_p = tf.keras.activations.softplus(tensorOut[...,price])
    out_q = tf.keras.activations.softplus(tensorOut[...,ir])
    return tf.concat([out_e,out_b,out_p,out_q],axis=-1)

inp = Input(shape = (T,input), name="input")
x = LSTM(units = 128, return_sequences=True, name="hidden1")(inp)
x = LSTM(units = 64, return_sequences=True, name="hidden2")(x)
out = Dense(units = output, activation = activation_final, name="ThisPeriod")(x)
forecast = Dense(units = outputF*S, activation='softplus')(x)
outF = Reshape((T,S,outputF), name="Forecast")(forecast)
model = Model(inputs=inp,outputs=[out,outF,inp])
model.summary()