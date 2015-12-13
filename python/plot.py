import numpy as np
import matplotlib.pyplot as plt

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
	IPython.embed()
	colors = {k:c for k,c in zip(mean_hits.keys(),['r','g','b'])}
	
	for k in hits:
		for run in range(hits[k].shape[0]):
			plt.plot(itr, hits[k][run, :], color=colors[k], alpha=0.1)
		plt.plot(itr, mean_hits[k], label=k, color=colors[k])
	
	plt.legend()
	plt.show()


	IPython.embed()

if __name__ == '__main__':
	cov_type_dir = osp.join(results_dir,'covertype/run4')
	hits_ct = get_expts_from_dir(cov_type_dir)
	plot_expts(hits_ct)
