import numpy as np


class MultivariateUniform:
	def __init__(self, d):
		self.d = d

	def sample(self, n):
		return np.reshape(np.random.uniform(-1, 1, n * self.d), (n, self.d))



class LocationScaleModel:

	def __init__(self, mean_f, std_f, quantile_f, cdf_f):
		self.mean_f = mean_f
		self.std_f = std_f
		self.quantile_f = quantile_f
		self.cdf_f = cdf_f

	def sample(self, x):
		n = np.shape(x)[0]
		v = np.reshape(np.random.uniform(0, 1, n), (n, 1))
		y = self.mean_f(x) + self.std_f(x) * self.quantile_f(v)
		return np.reshape(y, (n, 1))

	def cdf(self, x, y):
		return self.cdf_f((y - self.mean_f(x)) / (self.std_f(x) + 1e-9))



class UnivariateResponseSampler:
	def __init__(self, x_sampler, yx_sampler):
		self.x_sampler = x_sampler
		self.yx_sampler	= yx_sampler

	def sample(self, n):
		x = self.x_sampler.sample(n)
		y = self.yx_sampler.sample(x)
		return x, y

	def sample_multi_y(self, n, m):
		x = self.x_sampler.sample(n)
		ys = []
		for i in range(m):
			ys.append(np.reshape(self.yx_sampler.sample(x), (n, 1, 1)))
		return x, np.concatenate(ys, 1)


def model1():
	# set data generating process
	# Y = sin(X) + 0.3 (cos(X) + 1.5) U   with   U ~ uniform[-1, 1]
	# X

	def mean_func(x):
		return np.sin(3 * x)

	def std_func(x):
		return 1 - 0.7*np.cos(4*x)

	def quantile_func(x):
		return 2 * x - 1

	def cdf_func(x):
		return (x + 1.0) / 2
	
	modelx = MultivariateUniform(1)
	modelyx = LocationScaleModel(mean_func, std_func, quantile_func, cdf_func)
	return UnivariateResponseSampler(modelx, modelyx)