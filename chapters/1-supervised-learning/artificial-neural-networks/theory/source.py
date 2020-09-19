import nn
import mnist_loader


def run_nn(net, n_epoch, mini_batch_size, eta):
	train, validate, test = mnist_loader.load_data_wrapper()
	net.SGD_watch(train, n_epoch, mini_batch_size, eta, test)


if __name__ == "__main__":
	n_epoch, mini_batch_size, eta = 30, 20, 3.0
	# net1 = nn.Network([784, 30, 10])
	# run_nn(net1, n_epoch, mini_batch_size, eta)

	# net2 = nn.Network([784, 100, 10])
	# run_nn(net2, n_epoch, mini_batch_size, eta)

	net3 = nn.Network([784, 30, 70, 10])
	run_nn(net3, n_epoch, mini_batch_size, eta)


