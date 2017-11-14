# dependencies for code
from csv import reader
import numpy as np
import sys

# In this code, we are going to use the Pima Indians onset of diabetes dataset. It describes patient medical record data for Pima Indians and whether they had an onset of diabetes within five years.
with open('data.csv', 'r') as f:
	reader = reader(f)
	your_list = list(reader)
	dataset = np.array(your_list, dtype = float)

x_temp = dataset[:,0:8]
y_temp= dataset[:,8]

#converting y_temp list into column vector 
y = (y_temp.reshape(768,1))

#normalizing features
x = (x_temp - np.mean(x_temp, axis = 0))/ np.std(x_temp, axis = 0)

#activation function for non linearity
def sigmoid(X):
	return 1 / (1 + np.exp(-X))

def relu(X):
	return X * (X > 0)


#dervative of activation function (will be used in backprop) 
def sigmoid_out2deriv(out):
	return (out) * (1 - out)

def relu_out2deriv(out):
	return 1. * (out > 0)


#creating decoupled neural interfaces  class

class DNI(object):
	def __init__(self,input_dim, output_dim, nonlinear, nonlin_derivative, alpha):
		self.weights =np.random.normal(size = (input_dim, output_dim), scale=0.1) # weights for neural net

		#weights for generator (i'm adding 1 hidden layer in generator for better output)
		self.weights_synthetic_grads_0_1 = np.random.normal(size=(output_dim,32), scale=0.01)
		self.weights_synthetic_grads_1_2 = np.random.normal(size=(32, output_dim), scale=0.01)

		self.nonlin = nonlinear
		self.nonlin_deriv = nonlin_derivative
		self.alpha = alpha #learning rate


	def forward_and_synthetic_update(self,input):
		#forward pass of main neural net
		self.input = input
		self.output = self.nonlin(np.dot(self.input, self.weights))

		#forward pass of respective generator network
		self.synthetic_gradient_hidden = self.nonlin(np.dot(self.output, self.weights_synthetic_grads_0_1))
		self.synthetic_gradient = np.dot(self.synthetic_gradient_hidden, self.weights_synthetic_grads_1_2)

		#using output of generator as output delta for  respective layer in main neural net 
		self.weights_synthetic_gradient = self.synthetic_gradient * self.nonlin_deriv(self.output)

		# updating weights of main neural net
		self.weights -= self.input.T.dot(self.weights_synthetic_gradient) * self.alpha
		return np.dot(self.weights_synthetic_gradient, self.weights.T) , self.output # np.dot(self.weights_synthetic_gradient, self.weights.T) is true gradient for lower level generator

	def update_synthetic_weights(self,true_gradient):
		# using true gradient as label for generator  which we will get from backprop 
		self.synthetic_delta_2= self.synthetic_gradient - true_gradient 
		self.weights_synthetic_grads_1_2 -= self.synthetic_gradient_hidden.T.dot(self.synthetic_delta_2) * self.alpha

		self.synthetic_output_delta_1 = np.dot(self.synthetic_delta_2 , self.weights_synthetic_grads_1_2.T)
		self.synthetic_delta_1 = self.synthetic_output_delta_1 * self.nonlin_deriv(self.synthetic_gradient_hidden)
		self.weights_synthetic_grads_0_1 -= np.dot(self.output.T, self.synthetic_delta_1) * self.alpha 




#hyperparameters
input_dim = len(x[0])
layer_1_dim = 256
layer_2_dim = 128
output = len(y[0])
batch_size = 6
iterations =1500

#initilizing layer

layer_1 = DNI(input_dim,layer_1_dim,relu,relu_out2deriv,alpha= 0.01)
layer_2 = DNI(layer_1_dim,layer_2_dim,sigmoid,sigmoid_out2deriv,alpha = 0.01)
layer_3 = DNI(layer_2_dim, output,sigmoid, sigmoid_out2deriv,alpha = 0.01)

for iter in range(iterations):
	#creating minibatches
	for batch_i in range(int(len(x) / batch_size)):
		batch_x = x[(batch_i * batch_size):(batch_i+1)*batch_size]
		batch_y = y[(batch_i * batch_size):(batch_i+1)*batch_size]
		
		_,layer_1_out = layer_1.forward_and_synthetic_update(batch_x)
		layer_1_delta, layer_2_out = layer_2.forward_and_synthetic_update(layer_1_out)
		layer_2_delta, layer_3_out = layer_3.forward_and_synthetic_update(layer_2_out)

		layer_3_delta = layer_3_out - batch_y
		layer_3.update_synthetic_weights(layer_3_delta)
		layer_2.update_synthetic_weights(layer_2_delta)
		layer_1.update_synthetic_weights(layer_1_delta)


		error = np.mean((layer_3_out - batch_y)**2)
		synthetic_error =  np.mean((layer_3_delta - layer_3.synthetic_gradient)**2)

	if(iter % 100 == 10):
		sys.stdout.write("\rIter:" + str(iter) + " Loss:" + str(error) + " Synthetic Loss:" + str(synthetic_error))
	if(iter % 100 == 10):
		print("")
		print(batch_y, layer_3_out )




