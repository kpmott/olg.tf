from packages import *
from params import *

def activation_final(tensorOut):
    #τ = tf.shape(tensorOut)[0]
    out_c = tf.keras.activations.softplus(tensorOut[:,cons])
    out_e = tf.keras.activations.tanh(tensorOut[:,equity])
    out_b = tf.keras.activations.softplus(tensorOut[:,bond])
    out_p = tf.keras.activations.softplus(tensorOut[:,price])
    out_q = tf.keras.activations.softplus(tensorOut[:,ir])
    return tf.concat([out_c,out_e,out_b,out_p,out_q],axis=1)

model = Sequential()
model.add(Dense(128, input_dim=input,    activation='sigmoid')) # Hidden 1
model.add(Dense(128,                     activation='sigmoid')) # Hidden 2
model.add(Dense(128,                     activation='sigmoid')) # Hidden 3
model.add(Dense(output,                 activation=activation_final)) # Output
