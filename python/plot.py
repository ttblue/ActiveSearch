import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec

import os, os.path as osp
import cPickle as pick

import IPython

np.set_printoptions(suppress=True, precision=5, linewidth=100)

data_dir = os.getenv('AS_DATA_DIR')
results_dir = os.getenv('AS_RESULTS_DIR')

def get_expts_from_dir (dir_path):
	fnames = os.listdir(dir_path)

	expt_data = []
	for fname in fnames:
		with open(osp.join(dir_path,fname),'r') as fh: 
			expt_data.append(pick.load(fh))

	hits = {k:[] for k in expt_data[0].keys()}

	for dat in expt_data:
		for k in dat.keys():
			hits[k].append(dat[k])

	hits = {k:np.array(hits[k]) for k in hits.keys()}
	return hits

def plot_expts (hits):

	itr = range(hits[hits.keys()[0]].shape[1])
	mean_hits = {k:hits[k].mean(axis=0).squeeze() for k in hits} 
	mean2_hits = {k:hits[k].mean(axis=1).squeeze() for k in hits}
	max_hits = {k:hits[k].max(axis=1).squeeze() for k in hits}
	colors = {k:c for k,c in zip(mean_hits.keys(),['r','g','b'])}
	
	for k in hits:
		for run in range(hits[k].shape[0]):
			plt.plot(itr, hits[k][run, :], color=colors[k], alpha=0.1)
		plt.plot(itr, mean_hits[k], label=k, color=colors[k])
	
	plt.legend(loc=2)
	plt.show()

def plot_subplot_expts(hits):

	mean_hits = {k:hits[k].mean(axis=0).squeeze() for k in hits} 
	max_hits = {k:hits[k].max(axis=1).squeeze() for k in hits}
	colors = {k:c for k,c in zip(mean_hits.keys(),['r','g','b'])}
	itr = range(hits[hits.keys()[0]].shape[1])
	exp = range(10)
	max_alg = {i:max([max_hits[k][i] for k in max_hits]) for i in exp}
	overall_max = max(max_alg.values())

	gs = gridspec.GridSpec(5,5)
	
	ax = {}
	for i in exp:
		ax[i] = plt.subplot(gs[int(i>=5), i%5])
		ax[i].set_ylim([0,overall_max])
		ax[i].axes.get_xaxis().set_ticks([])
	ax[10] = plt.subplot(gs[2:,:])
	

	# IPython.embed()
	sub_itr = range(200, hits[hits.keys()[0]].shape[1])
	n_sub = len(sub_itr)
	for k in hits:
		for run in range(hits[k].shape[0]):
			ax[run].plot(sub_itr, hits[k][run, -n_sub:], color=colors[k], linewidth=3, alpha=0.8)
			ax[run].set_yticks([max_alg[run]])
		ax[10].plot(itr, mean_hits[k], label=k, color=colors[k], linewidth=5, alpha=0.8)
	
	plt.legend(loc=2)
	plt.show()


if __name__ == '__main__':
	import sys
	if len(sys.argv) > 1:
		expt_dir = osp.join(results_dir,sys.argv[1])
		if not osp.isdir(expt_dir):
			expt_dir = osp.join(results_dir,'covertype/run1')
	else:
		expt_dir = osp.join(results_dir,'covertype/run1')

	matplotlib.rcParams.update({'font.size': 20})

	hits = get_expts_from_dir(expt_dir)
	plot_subplot_expts(hits)
