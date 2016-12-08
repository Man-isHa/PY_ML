#Learning Sine wave
import theano
import numpy as np
import matplotlib.pyplot as plt
import theano.tensor as T
import math
theano.config.floatX = 'float64'


##  data 
step_radians = 0.01
steps_of_history = 200
steps_in_future = 1
index = 0

x = np.sin(np.arange(0, 20*math.pi, step_radians))

seq = []
next_val = []

for i in range(0, len(x)-steps_of_history, steps_of_history):
    seq.append(x[i: i + steps_of_history])
    next_val.append(x[i+1:i + steps_of_history+1])

seq = np.reshape(seq, [-1, steps_of_history, 1])
next_val = np.reshape(next_val, [-1,steps_of_history, 1])


trainX = np.array(seq)
trainY = np.array(next_val)


## model
n = 50
nin = 1
nout = 1

u = T.matrix()

t = T.matrix()

h0 = T.vector()
h_in = np.zeros(n).astype(theano.config.floatX)
lr = T.scalar()

W = theano.shared(np.random.uniform(size=(n, n), low=-.01, high=.01).astype(theano.config.floatX))
W_in = theano.shared(np.random.uniform(size=(nin, n), low=-.01, high=.01).astype(theano.config.floatX))
W_out = theano.shared(np.random.uniform(size=(n, nout), low=-.01, high=.01).astype(theano.config.floatX))


def step(u_t, h_tm1, W, W_in, W_out):	
	h_t = T.tanh(T.dot(u_t, W_in) + T.dot(h_tm1, W))
    	y_t = T.dot(h_t, W_out)
    	return h_t, y_t


[h, y], _ = theano.scan(step,
                        sequences=u,
                        outputs_info=[h0, None],
                        non_sequences=[W, W_in, W_out])

error = ((y - t) ** 2).sum()
prediction = y
gW, gW_in, gW_out = T.grad(error, [W, W_in, W_out])


fn = theano.function([h0, u, t, lr],
                     error,
                     updates={W: W - lr * gW,
                             W_in: W_in - lr * gW_in,
                             W_out: W_out - lr * gW_out})
predict = theano.function([h0, u], prediction)


#for e in range(10):
for i in range(len(trainX)):
	fn(h_in,trainX[i],trainY[i],0.001)
	
	
print('End of training')


x = np.sin(np.arange(20*math.pi, 24*math.pi, step_radians))

seq = []

for i in range(0, len(x)-steps_of_history, steps_of_history):
    seq.append(x[i: i + steps_of_history])

seq = np.reshape(seq, [-1, steps_of_history, 1])
testX = np.array(seq)

# Predict the future values
predictY = []
for i in range(len(testX)):
	predictY = predictY+predict(h_in,testX[i]).tolist()

print(predictY)
# Plot the results

plt.plot(x, 'r-', label='Actual')
plt.plot(np.asarray(predictY), 'gx', label='Predicted')

plt.show()
