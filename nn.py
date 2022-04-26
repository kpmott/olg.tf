from packages import *
from params import *

import detSS
ebar,bbar,pbar,qbar,xbar,cbar = detSS.detSS_allocs()

def activation_final(tensorOut):
    out_e = tf.keras.activations.tanh(tensorOut[...,equity])
    out_b = tf.keras.activations.tanh(tensorOut[...,bond])
    out_p = tf.keras.activations.softplus(tensorOut[...,price])
    out_q = tf.keras.activations.softplus(tensorOut[...,ir])
    return tf.concat([out_e,out_b,out_p,out_q],axis=-1)

inp = Input(shape=(input,))
x = Dense(units = 512,name='Hidden1',activation='relu')(inp[...,:-1]) #,kernel_initializer=initializers.Zeros(),bias_initializer=initializers.Zeros()
#x = Dense(units = 512,name='Hidden2',activation='relu')(x)
x = Dense(units = 256 ,name='Hidden3',activation='relu')(x)
outp = Dense(units = output-1, activation=activation_final,name='AssetsPrices')(x)
outpt = tf.concat([outp,tf.expand_dims(inp[...,-1],-1)],-1)

#mirrored_strategy = tf.distribute.MirroredStrategy()
#with mirrored_strategy.scope():
model = Model(inputs=inp,outputs=outpt)

model.summary()
plot_model(model, "model.png", show_shapes=True)