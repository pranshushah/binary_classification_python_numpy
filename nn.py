# dependencies for code
from csv import reader
import numpy as np
import sys


# In this code, we are going to use the Pima Indians onset of diabetes dataset. It describes patient medical record data for Pima Indians and whether they had an onset of diabetes within five years.
with open('data.csv', 'r') as f:
	reader = reader(f)
	your_list = list(reader)
	dataset = np.array(your_list, dtype = float)

x_temp= dataset[:,0:8]
y_temp= dataset[:,8]

#normalizing the data
x = (x_temp - np.mean(x_temp, axis = 0)) /np.std(x_temp, axis = 0)

#converting list into columnm vector
y = y_temp.reshape(768,1)


#activation function for non_linearity
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

#derivative of activation function (will be used during backprop) 
def sigmoid_out2deriv(out):
	return out * (1 - out)

class Layer(object):
	def __init__(self, input_dim, output_dim, nonlinear,nonlin_derivative, dropout = True, dropout_percent = 0.5):
		#initilizing constructor
		self.weights =np.random.normal(size = (input_dim, output_dim), scale=0.05) #initilizing weight from gussian distribution with 0 mean and 0.05 std
		self.nonlin = nonlinear
		self.nonlin_deriv = nonlin_derivative
		self.dropout = dropout
		self.dropout_percent = dropout_percent
	
	#forward pass
	def forward(self, input):
		self.input = input
		self.output = self.nonlin(np.dot(self.input,self.weights))
		#implementing dropout
		if(self.dropout):
			self.mask = (np.random.random_sample(self.output.shape))
			self.mask = self.mask * (self.mask >= self.dropout_percent)
			# Inverted dropout scales the remaining neurons during training so we don't have to at test time.
			self.mask = self.mask * 1/(1 - self.dropout_percent)
			self.output *= self.mask
		return self.output
	#backward pass
	def backward(self, output_delta):
		if(self.dropout):
			self.weight_delta = (output_delta) * self.mask * (self.nonlin_deriv(self.output))
		else:
			self.weight_delta= np.multiply(output_delta, self.nonlin_deriv(self.output))
		return np.dot(self.weight_delta,self.weights.T)
	# updating weights
	def update(self,alpha = 0.005):
		self.weights -= np.dot(self.input.T, self.weight_delta) * alpha


#hyperparameters
input_dim = len(x[0])
layer_1_dim = 128
layer_2_dim = 64
output_dim = 1
batch_size = 10


#initilizing layers
layer_1 = Layer(input_dim,layer_1_dim,sigmoid,sigmoid_out2deriv, dropout = False, dropout_percent = 0.5)
layer_2 = Layer(layer_1_dim,layer_2_dim,sigmoid,sigmoid_out2deriv, dropout = True, dropout_percent = 0.5)
layer_3 = Layer(layer_2_dim, output_dim,sigmoid, sigmoid_out2deriv, dropout = False, dropout_percent = 0.5)

iterations = 1000 #number of epochs

for iter in range(iterations):
    error = 0

    #creating mini_batches 
    for batch_i in range(int(len(x) / batch_size)):
        batch_x = x[(batch_i * batch_size):(batch_i+1)*batch_size]
        batch_y = y[(batch_i * batch_size):(batch_i+1)*batch_size]  
        
        layer_1_out = layer_1.forward(batch_x)
        layer_2_out = layer_2.forward(layer_1_out)
        layer_3_out = layer_3.forward(layer_2_out)

        layer_3_delta = layer_3_out - batch_y  #(gradient of mse)
        layer_2_delta = layer_3.backward(layer_3_delta)
        layer_1_delta = layer_2.backward(layer_2_delta)
        layer_1.backward(layer_1_delta)
        
        layer_1.update()
        layer_2.update()
        layer_3.update()
        
        error = np.mean((layer_3_out - batch_y)**2)

    sys.stdout.write("\rIter:" + str(iter) + " Loss:" + str(error))
    if(iter % 50 == 0):
        print("")
        print(layer_3_out)




		
