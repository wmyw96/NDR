from model import *
from data import *
import numpy as np



import os
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--seed", help="random seed", type=int, default=1234)
parser.add_argument("--n", help="number of samples", type=int, default=500)
parser.add_argument("--m1", help="number of noise samples per data", type=int, default=64)
parser.add_argument("--m2", help="number of noise samples per batch", type=int, default=2)

parser.add_argument("--dim_x", help="number of explanatory vars", type=int, default=1)

parser.add_argument("--nepochs", help="number of epoches", type=int, default=100)
parser.add_argument("--batch_size", help="batch size", type=int, default=64)
parser.add_argument("--lr", help="learning rate", type=float, default=1e-3)

parser.add_argument("--depth", help="neural network depth", type=int, default=2)
parser.add_argument("--width", help="neural network width", type=int, default=256)
parser.add_argument("--noise_width", help="neural network noise width", type=int, default=64)

parser.add_argument("--ex_id", help="example id", type=int, default=1)
args = parser.parse_args()


if args.ex_id == 1:
	dgp = model1()

	n = 1000
	x, y = dgp.sample(n)
	x2, y2_true = dgp.sample(2000)
	x2 *= 2

	ndr = NeuralDistributionalRegression(input_dim=1, input_noise_dim=1, out_dim=1, 
			depth=2, width=200, noise_width=100, noise_type='uniform')
	ndr.fit(x, y, num_epoches=500, noise_per_sample=1, batch_size=500, noise_per_batch=2)

	y2 = ndr.predict(x2, 100)
	#print(ndr.eval(dgp, (30, 100), 500))

	import matplotlib.pyplot as plt
	from matplotlib import rc
	from numpy import genfromtxt

	plt.rcParams["font.family"] = "Times New Roman"
	plt.rc('font', size=20)
	rc('text', usetex=True)

	color_tuple = [
		'#ae1908',  # red
		'#ec813b',  # orange
		'#05348b',  # dark blue
		'#9acdc4',  # pain blue
	]

	xx = np.arange(-1, 1, 0.01)
	plt.plot(xx, np.sin(3 * xx), color=color_tuple[0], linestyle='solid')

	plt.plot(xx, np.sin(3 * xx) + (1-0.7*np.cos(4*xx)), color=color_tuple[0], linestyle='dotted')
	plt.plot(xx, np.sin(3 * xx) - (1-0.7*np.cos(4*xx)), color=color_tuple[0], linestyle='dotted')

	plt.scatter(x2, y2[:, 0, 0], s=0.2, color=color_tuple[2])
	
	plt.scatter(x2, np.mean(y2[:, :, 0], 1), s=0.2, color=color_tuple[1])

	plt.ylim(-6, 6)
	plt.xlim(-2, 2)
	plt.show()

