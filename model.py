import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.optim as optim 


def pairwise_distance(x, y, offset):
	n1, m1, d1 = np.shape(x)[0], np.shape(x)[1], np.shape(x)[2]
	n2, m2, d2 = np.shape(y)[0], np.shape(y)[1], np.shape(y)[2]
	assert (n1 == n2) and (d1 == d2)
	x2 = np.tile(np.reshape(x, (n1, m1, 1, d1)), (1, 1, m2, 1))
	y2 = np.tile(np.reshape(y, (n2, 1, m2, d2)), (1, m1, 1, 1))
	return np.mean(np.mean(np.sqrt(np.sum(np.square(x2 - y2), 3) + 1e-9), (1, 2)) * m1 / (m1-offset))


def pairwise_distance_bf(x, y, scale):
	n1, m1, d1 = np.shape(x)[0], np.shape(x)[1], np.shape(x)[2]
	n2, m2, d2 = np.shape(y)[0], np.shape(y)[1], np.shape(y)[2]
	assert (n1 == n2) and (d1 == d2)
	dis = 0
	for i in range(m1):
		for j in range(m2):
			dis += np.mean(np.sqrt(np.sum(np.square(x[:, i, :] - y[:, j, :]), 1) + 1e-9)) / scale
	return dis
	#x2 = np.reshape(x, (n1, m1, 1, d1))
	#y2 = np.reshape(y, (n2, 1, m2, d2))
	


class RandomSeedGenerator:
	def __init__(self, cir_round, start=2024, bases=[999983, 4869], md=1000000007):
		self.cir_round = cir_round
		self.value = start
		self.start_value = start
		self.bases = bases
		self.md = md
		self.count = 0

	def next_value(self):
		self.count += 1
		if self.cir_round == self.count:
			self.count = 0
			self.value = self.start_value
		self.value = (self.value * self.bases[0] + self.bases[1]) % self.md
		return self.value



class StochasticReLUMLP(torch.nn.Module):

	def __init__(self, input_dim, input_noise_dim, out_dim, depth, width, noise_width, noise_type='uniform'):
		super(StochasticReLUMLP, self).__init__()

		self.input_dim = input_dim
		self.input_noise_dim = input_noise_dim
		self.out_dim = out_dim
		self.depth = depth
		self.width = width
		self.noise_width = noise_width
		if noise_type == 'normal':
			self.noise_dist = torch.distributions.normal.Normal(0, 1)
		else:
			# Here we use uniform[-sqrt(3), sqrt(3)] to keep unit variance
			self.noise_dist = torch.distributions.uniform.Uniform(-np.sqrt(3), np.sqrt(3))

		assert depth >= 1
		
		width_wonoise = width - noise_width
		self.layers = []
		self.layers.append(
				nn.Sequential(OrderedDict([('linear1', nn.Linear(input_dim + input_noise_dim, width_wonoise)), 
										   ('relu1', nn.ReLU())]))
			)

		for l in range(depth - 1):
			self.layers.append(
					nn.Sequential(OrderedDict([(f'linear{l+2}', nn.Linear(width, width_wonoise)), 
										   		(f'relu{l+2}', nn.ReLU())]))
				)

		self.layers.append(nn.Linear(width_wonoise, out_dim))


	def forward(self, x, noise_samples=2):
		'''
			Parameters
			----
				x : torch of size [batch_size, x_dim]

			Returns
			----
				a torch.tensor with shape [batch_size, noise_samples, y_dim]
		'''
		batch_size, x_dim = x.shape[0], x.shape[1]

		x_unsqueezed = x.unsqueeze(1)
		x_expanded = x_unsqueezed.expand(batch_size, noise_samples, x_dim)

		noise = self.noise_dist.sample((batch_size, noise_samples, self.input_noise_dim))
		z = torch.concatenate([x_expanded, noise], 2)

		#print(f'NDR forward, input layer = {z.shape}')
		for l in range(self.depth):
			z = self.layers[l](z)
			noise = self.noise_dist.sample((batch_size, noise_samples, self.noise_width))
			if l + 1 < self.depth:
				z = torch.concatenate([z, noise], 2)
			#print(f'NDR forward, hidden layer {l} = {z.shape}')


		return self.layers[self.depth](z)

	def parameters(self):
		params = []
		for l in range(self.depth + 1):
			for para in self.layers[l].parameters():
				params.append(para)
		return params

	def parameters_enumer(self):
		for l in range(self.depth + 1):
			for para in self.layers[l].parameters():
				print(f'Layer {l}, Parameter Shape = {para.shape}')



def torchl2norm(x):
	#return torch.squeeze(torch.abs(x), -1)
	return torch.sqrt(torch.sum(torch.square(x), -1) + 1e-9)


class NeuralDistributionalRegression:

	def __init__(self, input_dim, input_noise_dim, out_dim, depth, width, noise_width, noise_type, log=False):
		self.model = StochasticReLUMLP(input_dim, input_noise_dim, out_dim, depth, width, noise_width, noise_type)
		self.fitted = False
		if log:
			print(f'--- Neural Distributional Regression ---')
			self.model.parameters_enumer()

	def fit(self, feature, response, num_epoches=100, noise_per_sample=100, batch_size=128, noise_per_batch=2, 
			learning_rate=1e-2, weight_decay=0, log=False):
		optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
		seed_gen = RandomSeedGenerator(noise_per_sample)
		n = np.shape(feature)[0]

		if log:
			print(f'Fit: number of samples (n) = {n}, m_1 = {noise_per_sample}, m_2 = {noise_per_batch}, batch size = {batch_size}')

		loss_rec = []
		for epoch in range(num_epoches):
			seed = seed_gen.next_value()

			torch.manual_seed(seed)

			x_torch, y_torch = torch.tensor(feature).float(), torch.tensor(response).float()
			idx = torch.randperm(n)
			x_torch, y_torch = x_torch[idx, :], y_torch[idx, :]
			#print(f'data size: x {x_torch.shape}, y {y_torch.shape}')

			losses = []
			for it in range(n // batch_size):
				optimizer.zero_grad()
				self.model.train()

				x_batch = x_torch[it * batch_size: (it + 1) * batch_size, :]
				y_batch = y_torch[it * batch_size: (it + 1) * batch_size, :]
				#print(f'batch size: x {x_batch.shape}, y {y_batch.shape}')

				y_batch = y_batch[:, None, :]
				y_batch = y_batch.expand((batch_size, noise_per_batch, self.model.out_dim))

				y_pred = self.model(x_batch, noise_per_batch)

				pairwise_diff = (y_pred[:, :, None, :]).expand((batch_size, noise_per_batch, noise_per_batch, self.model.out_dim)) \
								- (y_pred[:, None, :, :]).expand((batch_size, noise_per_batch, noise_per_batch, self.model.out_dim))
				#print(f'pairwise diffence matrix = {pairwise_diff.shape}')

				energy_diff = torch.sum(torchl2norm(y_batch - y_pred), [1]) / noise_per_batch 
				energy_self = torch.sum(torchl2norm(pairwise_diff), [1, 2]) / (noise_per_batch * (noise_per_batch - 1))

				eloss_diff = torch.mean(energy_diff)
				eloss_self = torch.mean(energy_self)

				loss = 2 * eloss_diff - eloss_self 
				losses.append([loss.item(), eloss_diff.item(), eloss_self.item()])

				loss.backward()
				optimizer.step()

			loss_rec.append(np.mean(np.array(losses), 0))
			if log and ((epoch + 1) % (epoch // 7) == 0):
				print(f'Epoch {epoch}, loss = {loss_rec[-1][0]}, eloss_diff = {loss_rec[-1][1]}, eloss_self = {loss_rec[-1][2]}')

		self.fitted = True


	def predict(self, x, num_noise_samples):
		x_torch = torch.tensor(x).float()
		self.model.eval()
		return self.model(x_torch, num_noise_samples).detach().cpu().numpy()


	def eval(self, sampler, n, m):
		losses = []
		self.model.eval()
		for i in range(n[0]):
			x, y = sampler.sample_multi_y(n[1], m)
			y2 = self.model(torch.tensor(x).float(), m).detach().cpu().numpy()
			loss = 2*pairwise_distance(y, y2, 0) - pairwise_distance(y, y, -1) - pairwise_distance(y2, y2, -1)
			losses.append(loss)
		return np.mean(losses)



















































