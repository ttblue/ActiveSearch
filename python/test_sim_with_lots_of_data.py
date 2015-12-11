from __future__ import division
import numpy as np, numpy.random as nr, numpy.linalg as nlg
import scipy as sp, scipy.linalg as slg, scipy.io as sio, scipy.sparse as ss
import matplotlib.pyplot as plt
import time
import os, os.path as osp
import csv

import adaptiveActiveSearch as AAS
import activeSearchInterface as ASI
import similarityLearning as SL

np.set_printoptions(suppress=True, precision=5, linewidth=100)

data_dir = osp.join('/home/sibiv',  'Research/Data/ActiveSearch/Kyle/data/KernelAS')

def load_covertype (sparse=False):

	fname = osp.join(data_dir, 'covtype.data')
	fn = open(fname)
	data = csv.reader(fn)

	r = 54

	classes = []
	if sparse:
		Y = []
		rows = []
		cols = []
		sdat = []

		c = 0
		for line in data:
			y = int(line[-1])
			Y.append(y)
			if y not in classes: classes.append(y)

			xvec = np.array(line[:54]).astype(float)
			xcol = xvec.nonzero()[0].tolist()

			rows.extend(xcol)
			cols.extend([c]*len(xcol))
			sdat.extend(xvec[xcol].tolist())

			c += 1

		X = ss.csr_matrix((sdat, (rows, cols)), shape=(r, c))

	else:

		X = []
		Y = []
		for line in data:
			X.append(np.asarray(line[:54]).astype(float))
			y = int(line[-1])
			Y.append(y)
			if y not in classes: classes.append(y)

		X = np.asarray(X).T

	fn.close()

	Y = np.asarray(Y)
	return X, Y, classes

def stratified_sample (X, Y, classes, strat_frac=0.1):

	inds = []
	for c in classes:
		c_inds = (Y==c).nonzero()[0]
		c_num = int(len(c_inds)*strat_frac)
		inds.extend(c_inds[nr.permutation(len(c_inds))[:c_num]].tolist())

	Xs = X[:,inds]
	Ys = Y[inds]

	return Xs, Ys



def test_covtype ():

	nr.seed(0)

	verbose = True
	sparse = True
	pi = 0.5
	eta = 0.7
	K = 1000
	T = 1000

	sl_alpha = 0.001
	sl_C = 0.001
	sl_gamma = 0.01
	sl_margin = 1.
	sl_sampleR = 5000
	sl_epochs = 50;
	sl_npairs_per_epoch = 20000
	sl_nneg_per_pair = 1
	

	n_samples_pos = 300;
	n_samples_neg = 20000;

	strat_frac = 0.2
	X0,Y0,classes = load_covertype(sparse=sparse)
	X, Y = stratified_sample(X0, Y0, classes, strat_frac=strat_frac)

	d,n = X.shape

	X_norms = np.sqrt(((X.multiply(X)).sum(axis=0))).A.squeeze()
	X = X.dot(ss.spdiags([1/X_norms],[0],n,n)) # Normalization

	cl = 4
	Y = (Y==cl)

	W0 = np.eye(d)
	print 'Loaded the data'
	
	init_pt_pos = Y.nonzero()[0][nr.choice(len(Y.nonzero()[0]),n_samples_pos,replace=False)]
	print 'Sampled the positive data'

	init_pt_neg = (Y == 0).nonzero()[0][nr.choice(len((Y == 0).nonzero()[0]),n_samples_neg,replace=False)]
	print 'Sampled Negative Data'

	idxs = np.concatenate((init_pt_pos, init_pt_neg));
	X_sampled = X[:,idxs]
	Y_sampled = Y[idxs]

	prms = ASI.Parameters(pi=pi,sparse=sparse, verbose=True, eta=eta)
	slprms = SL.SPSDParameters(alpha=sl_alpha, C=sl_C, gamma=sl_gamma, margin=sl_margin, 
		epochs=sl_epochs, npairs_per_epoch=sl_npairs_per_epoch, nneg_per_pair=sl_nneg_per_pair)

	kAS = ASI.kernelAS (prms)
	aAS = AAS.adaptiveKernelAS(W0, T, prms, slprms)

	# kAS.initialize(X,init_labels={init_pt:1})
	aAS.initialize(X,init_labels={})

	# Now learn the similarity using the sampled data
	aAS.spsdSL.initialize(X_sampled,Y_sampled,W0,slprms)
	aAS.spsdSL.runSPSD()
	W0 = aAS.spsdSL.getW()

	# ---------------------------------------------
	## Now we reinitialize everything so that we can use the above learnt similarity 
	# to do active search
	init_pt = Y.nonzero()[0][nr.choice(len(Y.nonzero()[0]),2,replace=False)]

	prms = ASI.Parameters(pi=pi,sparse=sparse, verbose=True, eta=eta)
	slprms = SL.SPSDParameters(alpha=sl_alpha, C=sl_C, gamma=sl_gamma, margin=sl_margin, 
		epochs=sl_epochs, npairs_per_epoch=sl_npairs_per_epoch, nneg_per_pair=sl_nneg_per_pair)

	kAS = ASI.kernelAS (prms)
	aAS = AAS.adaptiveKernelAS(W0, T, prms, slprms)

	# kAS.initialize(X,init_labels={init_pt:1})
	aAS.initialize(X,init_labels={p:1 for p in init_pt})

	hits1 = [1]
	hits2 = [1]
	for i in xrange(K):

		# idx1 = kAS.getNextMessage()
		idx2 = aAS.getNextMessage()

		# kAS.setLabelCurrent(Y[idx1])
		aAS.setLabelCurrent(Y[idx2])

		# hits1.append(hits1[-1]+Y[idx1])
		hits2.append(hits2[-1]+Y[idx2])

	num_hits = hits2[-1];

	fileName = 'results_C_%.2f_gamma_%.4f.npy'%(sl_C,sl_gamma)
	fname = osp.join('../cvx_results', fileName)
	np.save(fname, num_hits)	

	fileName = 'sim_C_%.2f_gamma_%.4f.npy'%(sl_C,sl_gamma)
	fname = osp.join('../cvx_results', fileName)
	np.save(fname, W0)	
	
	import IPython
	IPython.embed()


if __name__ == '__main__':
	test_covtype()
