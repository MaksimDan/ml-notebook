import numpy as np
import progressbar as pb
import random


# Miscellaneous functions
def sigmoid(z):
	"""
	objective: compute sigmoid of input
	"""
	return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
	"""
	objective: compute sigmoid derivative of input
	"""
	return sigmoid(z)*(1-sigmoid(z))


class Network(object):
	def __init__(self, sizes):
		"""
		objective: initialize network architecture
		:param sizes: List[int] - number of neurons for each layer (including input layer)
		"""
		self.n_layers = len(sizes)
		self.sizes = sizes

		# using normal dist. mean=0, sd=1 to add a bias for every nueron,
		# except for the input layer
		self.biases = [np.random.randn(s, 1) for s in sizes[1:]]

		# normal dist. mean=0, sd=1, weight matrix vector for each
		# previous layer
		self.weights = [np.random.randn(forward, prev) for (forward,
						prev) in zip(sizes[1:], sizes[:-1])]

	def feed_forward(self, a):
		"""
		objective: compute output of a single neuron given input
		:param a: float - input of the network
		:return: int - next layer output
		"""
		for (b, w) in zip(self.biases, self.weights):
			a = sigmoid(np.dot(w, a) + b)
		return a

	def SGD_watch(self, train, epochs, mini_batch_size, eta, test):
		"""
		objective: stochastic gradient descent, our main learning algorithm
					that adjusts the weights, (or trains the neural network)
					It is also built to watch the progress of each epoch
		:param train: List[(List[float], float)] - training set (input, output)
		:param epochs: int - number of times to iterate over the training data
		:param mini_batch_size: int - random sample size for gradient descent
		:param eta: float - learning rate
		:param test: List[(List[float], float)] - validation set (input, output)
		:return:
		"""
		# ensure that the inputs match the first layer
		assert(all(len(image) == self.sizes[0] for image, _ in train))
		n_test = len(test)
		n = len(train)

		for j in range(epochs):
			# split the training data into separate random mini-batches
			random.shuffle(train)
			mini_batches = [
				train[k:k+mini_batch_size]
				for k in range(0, n, mini_batch_size)]

			# for each mini-batch, apply a single set of gradient descent
			# via update_mini_batch
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch, eta)

			# evaluate how well we are doing after each epoch
			print(f"Epoch {j}: {self.evaluate(test)} / {n_test}")

	def SGD(self, train, epochs, mini_batch_size, eta):
		"""
		objective: stochastic gradient descent, our main learning algorithm
					that adjusts the weights, (or trains the neural network)
		:param train: List[(List[float], float)] - training set (input, output)
		:param epochs: int - number of times to iterate over the training data
		:param mini_batch_size: int - random sample size for gradient descent
		:param eta: float - learning rate
		:return:
		"""
		n = len(train)
		for _ in range(epochs):
			random.shuffle(train)
			mini_batches = [
				train[k:k+mini_batch_size]
				for k in range(0, n, mini_batch_size)]
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch, eta)

	def update_mini_batch(self, mini_batch, eta):
		"""
		objective: applies a single step of gradient descent using
					back-propagation
		:param mini_batch: List[(List(float, int)] - training batch (input, output)
		:param eta: float - learning rate
		:return:
		"""
		# initialize the same exact data structures as the biases and weights
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]

		# compute the gradient for every mini-training batch
		for x, y in mini_batch:
			delta_nabla_b, delta_nabla_w = self.backprop(x, y)
			# add the results of the change from back propagation
			nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

		# update the weights and biases accordingly
		self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
		self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

	def backprop(self, x, y):
		"""
		objective: return the gradient associated with the cost of training sample x
		:param x: List[float] - input
		:param y: int - expected output
		:return: (List[nable_b], List[Matrix[nabla_w]]) - the gradient for the cost function C_x
		"""
		# initialize empty biases and weights
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]

		# feedforward
		activation = x
		activations = [x] # list to store all the activations, layer by layer
		zs = [] # list to store all the z vectors, layer by layer
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation)+b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)

		# backward pass
		delta = self.cost_derivative(activations[-1], y) * \
			sigmoid_prime(zs[-1])
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())
		# Note that the variable l in the loop below is used a little
		# differently to the notation in Chapter 2 of the book.  Here,
		# l = 1 means the last layer of neurons, l = 2 is the
		# second-last layer, and so on.  It's a renumbering of the
		# scheme in the book, used here to take advantage of the fact
		# that Python can use negative indices in lists.
		for l in range(2, self.n_layers):
			z = zs[-l]
			sp = sigmoid_prime(z)
			delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
		return nabla_b, nabla_w

	def evaluate(self, test_data):
		"""Return the number of test inputs for which the neural
		network outputs the correct result. Note that the neural
		network's output is assumed to be the index of whichever
		neuron in the final layer has the highest activation."""
		test_results = [(np.argmax(self.feed_forward(x)), y)
						for (x, y) in test_data]
		return sum(int(x == y) for (x, y) in test_results)

	def cost_derivative(self, output_activations, y):
		"""Return the vector of partial derivatives \partial C_x /
		\partial a for the output activations."""
		return output_activations-y
