def SGD_watch(self, training_data, epochs, mini_batch_size, eta, test_data):
	"""
	objective: stochastic gradient descent, our main learning algorithm
				that adjusts the weights, (or trains the neural network)
				It is also built to watch the progress of each epoch
	:param training_data: List[(List[float], float)] - training set (input, output)
	:param epochs: int - number of times to iterate over the training data
	:param mini_batch_size: int - random sample size for gradient descent
	:param eta: float - learning rate
	:param test_data: List[(List[float], float)] - validation set (input, output)
	:return:
	"""
	# ensure that the inputs match the first layer
	assert(all(len(image) == self.sizes[0] for image, _ in training_data))
	n_test = len(test_data)
	n = len(training_data)

	bar = pb.ProgressBar()
	for j in bar(range(epochs)):
		# split the training data into separate random mini-batches
		random.shuffle(training_data)
		mini_batches = [
			training_data[k:k+mini_batch_size]
			for k in range(0, n, mini_batch_size)]

		# for each mini-batch, apply a single set of gradient descent
		# via update_mini_batch
		for mini_batch in mini_batches:
			self.update_mini_batch(mini_batch, eta)
		# evaluate how well we are doing after each epoch

	if test_data:
		print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
	else:
		print(f"Epoch {j} complete")

