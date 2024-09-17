from model import *
from data import *
import numpy as np



import os
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--n", help="sample size", type=int, default=500)
parser.add_argument("--r", help="number of repeats", type=int, default=5)
parser.add_argument("--ex_id", help="example id", type=int, default=1)
args = parser.parse_args()

n = args.n
num_repeat = args.r


if args.ex_id == 1:
	m1_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
	m2_list = [2, 4, 8, 32]

	table = []
	for m1 in m1_list:
		for m2 in m2_list:
			for seed in range(num_repeat):
				dgp = model1()
				np.random.seed(seed)
				torch.manual_seed(seed)
				x, y = dgp.sample(n)

				ndr = NeuralDistributionalRegression(input_dim=1, input_noise_dim=1, out_dim=1, 
						depth=2, width=200, noise_width=0, noise_type='uniform')
				ndr.fit(x, y, num_epoches=500, noise_per_sample=m1, batch_size=500, noise_per_batch=m2)

				torch.manual_seed(seed)
				error = ndr.eval(dgp, (30, 100), 500)
				print(f'm1={m1}, m2={m2}, seed={seed}, error={error}')
				table.append((m1, m2, seed, error))
				
				np.savetxt(f"results/frontier_{n}_{num_repeat}.csv", np.array(table), delimiter=",")

elif args.ex_id == 2:
	# plot lib configuration

	#m1_list = [1, 2, 5, 10, 20, 50, 100, 200, 500]
	#m2_list = [2, 3, 4, 5, 7, 8, 10, 20, 30]
	m1_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
	m2_list = [2, 4, 8, 32]

	import matplotlib.pyplot as plt
	from matplotlib import rc

	plt.rcParams["font.family"] = "Times New Roman"
	plt.rc('font', size=20)
	rc('text', usetex=True)

	# end of configuration

	table = np.genfromtxt(f'results/frontier_{n}_{num_repeat}.csv', delimiter=',')
	from matplotlib import colors
	from matplotlib.colors import LinearSegmentedColormap, LogNorm
	color1 = "#ec813b"  # myred
	color2 = "#9acdc4"  # myblue

	cmap = LinearSegmentedColormap.from_list("custom_cmap", [color1, color2])

	m1 = []
	m2 = []
	val = []
	matrix = np.zeros((len(m1_list), len(m2_list)))
	for m1_id in range(len(m1_list)):
		for m2_id in range(len(m2_list)):
			idxes = m1_id * len(m2_list) * num_repeat + m2_id * num_repeat
			subtable = table[idxes:idxes+num_repeat, :]

			if idxes+num_repeat <= np.shape(table)[0]:
				m1.append(m1_list[m1_id])
				m2.append(m2_list[m2_id])
				val.append(np.mean(subtable[:, 3]))
				matrix[m1_id, m2_id] = np.mean(subtable[:, 3])
				print(f'm1={m1_list[m1_id]}, m2={m2_list[m2_id]}, idxes={idxes}, '+
					f'matched error = {np.sum(np.square(subtable[:, 0] - m1_list[m1_id]) + np.square(subtable[:, 1] - m2_list[m2_id]))}')
	
	#norm = LogNorm(vmin=min(val), vmax=max(val))
	norm = colors.Normalize(vmin=0, vmax=1)
	#plt.scatter(m1, m2, c=val, cmap=cmap, norm=norm)
	#plt.yscale('log')
	plt.xscale('log')
	#plt.colorbar()
	#plt.show()
	cand_plots = [0, 1, 2, 3]
	for i in cand_plots:
		plt.plot(m1_list, matrix[:, i], c=cmap((i+0.0)/len(cand_plots)), label=r'$m_2$' + f'={m2_list[i]}')
	plt.legend()
	plt.show()

elif args.ex_id == 3:
	#m1_list = [1, 2, 5, 10, 20, 50, 100, 200, 500]
	#m2_list = [2, 3, 4, 5, 7, 8, 10, 20, 30]
	m1_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
	m2_list = [2, 4, 8, 32]

	import matplotlib.pyplot as plt
	from matplotlib import rc

	plt.rcParams["font.family"] = "Times New Roman"
	plt.rc('font', size=20)
	rc('text', usetex=True)

	# end of configuration

	table = np.genfromtxt(f'results/frontier_{n}_{num_repeat}.csv', delimiter=',')
	from matplotlib import colors
	from matplotlib.colors import LinearSegmentedColormap, LogNorm
	color1 = "#ae1908"  # myred
	color2 = "#05348b"  # myblue

	cmap = LinearSegmentedColormap.from_list("custom_cmap", [color1, color2])

	m1 = []
	m2 = []
	m1m2 = []
	val = []
	matrix = np.zeros((len(m1_list), len(m2_list)))
	for m1_id in range(len(m1_list)):
		for m2_id in range(len(m2_list)):
			idxes = m1_id * len(m2_list) * num_repeat + m2_id * num_repeat
			subtable = table[idxes:idxes+num_repeat, :]

			if idxes+num_repeat <= np.shape(table)[0]:
				m1.append(m1_list[m1_id])
				m2.append(m2_list[m2_id])
				m1m2.append(m1_list[m1_id] * m2_list[m2_id])
				val.append(np.mean(subtable[:, 3]))
				matrix[m1_id, m2_id] = np.mean(subtable[:, 3])
				print(f'm1={m1_list[m1_id]}, m2={m2_list[m2_id]}, idxes={idxes}, '+
					f'matched error = {np.sum(np.square(subtable[:, 0] - m1_list[m1_id]) + np.square(subtable[:, 1] - m2_list[m2_id]))}')
	
	norm = colors.Normalize(vmin=0, vmax=1)
	#plt.scatter(m1, m2, c=val, cmap=cmap, norm=norm)
	plt.yscale('log')
	plt.xscale('log')
	#plt.colorbar()
	#plt.show()
	for i in range(4):
		plt.plot(np.array(m1_list) * m2_list[i], matrix[:, i], c=cmap((i+0.0)/4), label=r'$m_2$' + f'={m2_list[i]}')
	plt.legend()
	plt.show()


	plt.show()