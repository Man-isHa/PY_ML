import numpy as np
import theano as theano
import theano.tensor as T
import operator

'''
x = T.ones(5)
a=T.ones(1)
def sum1(x_t,s_prev,a):
	s_prev += x_t*a
	a=T.zeros(1)
	return s_prev,a
[s,a] ,updates = theano.scan(
		sum1,
		sequences = [x],
		outputs_info=[dict(initial = T.zeros(1)),a],
		)

'''

class RNNTheano:
    
    def __init__(self, word_dim, hidden_dim=256, dim_image=4096, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
	self.dim_image = dim_image
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
	E = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        U = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (6, hidden_dim, hidden_dim))
	Ui = np.random.uniform(-np.sqrt(1./dim_image), np.sqrt(1./dim_image), (dim_image))
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (6, hidden_dim, hidden_dim))
	bh = np.zeros((6,hidden_dim,))
	by = np.zeros((word_dim,))
        # Theano: Created shared variables
	self.E = theano.shared(name='E', value=E.astype(theano.config.floatX))
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
	self.Ui = theano.shared(name='Ui', value=Ui.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))    
	self.bh= theano.shared(name='bh', value=bh.astype(theano.config.floatX))  
	self.by= theano.shared(name='by', value=by.astype(theano.config.floatX))  

	# SGD / rmsprop: Initialize parameters
        self.mE = theano.shared(name='mE', value=np.zeros(E.shape).astype(theano.config.floatX))
        self.mU = theano.shared(name='mU', value=np.zeros(U.shape).astype(theano.config.floatX))
        self.mV = theano.shared(name='mV', value=np.zeros(V.shape).astype(theano.config.floatX))
        self.mW = theano.shared(name='mW', value=np.zeros(W.shape).astype(theano.config.floatX))
        self.mb = theano.shared(name='mb', value=np.zeros(bh.shape).astype(theano.config.floatX))
	self.mc = theano.shared(name='mc', value=np.zeros(by.shape).astype(theano.config.floatX))
	
        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()
    
    def __theano_build__(self):
        E, U, Ui, V, W ,bh ,by =self.E, self.U, self.Ui, self.V, self.W, self.bh, self.by
        x = T.ivector('x')
	img =T.dvector('img')        
	y = T.ivector('y')
	
	a = T.ones(1, dtype='float64')
        def forward_prop_step(x_t, s_t1_prev, s_t2_prev, a,E, U, Ui, V, W, bh, by ):
	    x_e = E[:,x_t]
            
            # GRU Layer 1
            z_t1 = T.nnet.hard_sigmoid(U[0].dot(x_e) + W[0].dot(s_t1_prev) + bh[0] + a * (Ui.dot(img)))
            r_t1 = T.nnet.hard_sigmoid(U[1].dot(x_e) + W[1].dot(s_t1_prev) + bh[1] + a * (Ui.dot(img)))
            c_t1 = T.tanh(U[2].dot(x_e) + W[2].dot(s_t1_prev * r_t1) + bh[2] + a * (Ui.dot(img)))
            s_t1 = (T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev
            
            # GRU Layer 2
            z_t2 = T.nnet.hard_sigmoid(U[3].dot(s_t1) + W[3].dot(s_t2_prev) + bh[3] + a * (Ui.dot(img)))
            r_t2 = T.nnet.hard_sigmoid(U[4].dot(s_t1) + W[4].dot(s_t2_prev) + bh[4] + a * (Ui.dot(img)))
            c_t2 = T.tanh(U[5].dot(s_t1) + W[5].dot(s_t2_prev * r_t2) + bh[5] + a * (Ui.dot(img)))
	    s_t2 = (T.ones_like(z_t2) - z_t2) * c_t2 + z_t2 * s_t2_prev
	    o_t = T.nnet.softmax(V.dot(s_t2) + by)[0]
	    a = T.zeros(1, dtype='float64')
            return [o_t, s_t1, s_t2, a]

        [o,s,s2,a], updates = theano.scan(
            	forward_prop_step,
            	sequences=[x],
            	outputs_info=[None, dict(initial=T.zeros(self.hidden_dim).astype('float64')), dict(initial=T.zeros(self.hidden_dim).astype('float64')),  a],
            	non_sequences=[E, U, Ui, V, W, bh,by],
            	truncate_gradient=self.bptt_truncate,
            	strict=True)
        
        prediction = T.argmax(o, axis=1)
        o_error = T.sum(T.nnet.categorical_crossentropy(o, y))
        
	cost = o_error
        # Gradients
	
	dE = T.grad(cost, E)
        dU = T.grad(o_error, U)
	dUi = T.grad(o_error, Ui)
        dV = T.grad(o_error, V)
        dW = T.grad(o_error, W)
	dbh = T.grad(o_error, bh)      
	dby = T.grad(o_error, by)      
	
        # Assign functions
        self.forward_propagation = theano.function([x,img], o)
        self.predict = theano.function([x,img], prediction)
	self.predict_class = theano.function([x,img], prediction)
        self.ce_error = theano.function([x,img, y], o_error)
        self.bptt = theano.function([x, img, y], [dE, dU, dUi, dV, dW, dbh, dby])
        
        # SGD
        learning_rate = T.scalar('learning_rate')
	decay = T.scalar('decay')

	        
        # rmsprop cache updates
        mE = decay * self.mE + (1 - decay) * dE ** 2
        mU = decay * self.mU + (1 - decay) * dU ** 2
        mW = decay * self.mW + (1 - decay) * dW ** 2
        mV = decay * self.mV + (1 - decay) * dV ** 2
        mb = decay * self.mb + (1 - decay) * dbh ** 2
	mc = decay * self.mc + (1 - decay) * dby ** 2

	self.sgd_step = theano.function(
            [x,img, y, learning_rate, theano.Param(decay, default=0.9)],
            [], 
            updates=[(E, E - learning_rate * dE / T.sqrt(mE + 1e-6)),
                     (U, U - learning_rate * dU / T.sqrt(mU + 1e-6)),
                     (W, W - learning_rate * dW / T.sqrt(mW + 1e-6)),
                     (V, V - learning_rate * dV / T.sqrt(mV + 1e-6)),
                     (bh, bh - learning_rate * dbh / T.sqrt(mb + 1e-6)),
                     (by, by - learning_rate * dby / T.sqrt(mc + 1e-6)),
                     (self.mE, mE),
                     (self.mU, mU),
                     (self.mW, mW),
                     (self.mV, mV),
                     (self.mb, mb),
                     (self.mc, mc)
	           ])
       
    
    def calculate_total_loss(self, X, img, Y):
        return np.sum([self.ce_error(x,img,y) for x,img, y in zip(X,img,Y)])
    
    def calculate_loss(self, X,img, Y):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X,img,Y)/float(num_words)   





def gradient_check_theano(model, x, y, h=0.001, error_threshold=0.01):
    # Overwrite the bptt attribute. We need to backpropagate all the way to get the correct gradient
    model.bptt_truncate = 1000
    # Calculate the gradients using backprop
    bptt_gradients = model.bptt(x, y)
    # List of all parameters we want to chec.
    model_parameters = ['U', 'V', 'W']
    # Gradient check for each parameter
    for pidx, pname in enumerate(model_parameters):
        # Get the actual parameter value from the mode, e.g. model.W
        parameter_T = operator.attrgetter(pname)(model)
        parameter = parameter_T.get_value()
        print "Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape))
        # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
        it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            # Save the original value so we can reset it later
            original_value = parameter[ix]
            # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
            parameter[ix] = original_value + h
            parameter_T.set_value(parameter)
            gradplus = model.calculate_total_loss([x],[y])
            parameter[ix] = original_value - h
            parameter_T.set_value(parameter)
            gradminus = model.calculate_total_loss([x],[y])
            estimated_gradient = (gradplus - gradminus)/(2*h)
            parameter[ix] = original_value
            parameter_T.set_value(parameter)
            # The gradient for this parameter calculated using backpropagation
            backprop_gradient = bptt_gradients[pidx][ix]
            # calculate The relative error: (|x - y|/(|x| + |y|))
            relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
            # If the error is to large fail the gradient check
            if relative_error > error_threshold:
                print "Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix)
                print "+h Loss: %f" % gradplus
                print "-h Loss: %f" % gradminus
                print "Estimated_gradient: %f" % estimated_gradient
                print "Backpropagation gradient: %f" % backprop_gradient
                print "Relative Error: %f" % relative_error
                return 
            it.iternext()
	print "Gradient check for parameter %s passed." % (pname)









