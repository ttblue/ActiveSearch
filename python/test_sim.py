from __future__ import division
import numpy as np, numpy.random as nr, numpy.linalg as nlg
import scipy as sp, scipy.linalg as slg, scipy.io as sio
import scipy.sparse as ss, scipy.sparse.linalg as sslg
import matplotlib.pyplot as plt
import time
import os, os.path as osp
import csv

import adaptiveActiveSearch as AAS
import activeSearchInterface as ASI
import similarityLearning as SL

np.set_printoptions(suppress=True, precision=5, linewidth=100)

data_dir = osp.join(os.getenv('HOME'),  'Research/Data/ActiveSearch/Kyle/data/KernelAS')

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

def return_average_positive_neighbors (X, Y, k):
	Y = np.asarray(Y)

	pos_inds = Y.nonzero()[0]
	Xpos = X[:,pos_inds]

	posM = np.array(Xpos.T.dot(X).todense())
	MsimInds = posM.argsort(axis=1)[:,-k:]
	MsimY =	Y[MsimInds]

	return MsimY.sum(axis=None)/(len(pos_inds)*k)


def test_covtype ():

	nr.seed(0)

	verbose = True
	sparse = True
	pi = 0.5
	eta = 0.7
	K = 1000
	T = 20

	sl_alpha = 0.001
	sl_C = 1e-10
	sl_gamma = 1e-10
	sl_margin = 1.
	sl_sampleR = 5000

	
	strat_frac = 0.1
	X0,Y0,classes = load_covertype(sparse=sparse)
	X, Y = stratified_sample(X0, Y0, classes, strat_frac=strat_frac)
	cl = 4
	Y = (Y==cl)
	d,n = X.shape
	
	X_norms = np.sqrt(((X.multiply(X)).sum(axis=0))).A.squeeze()
	X_normalized = X.dot(ss.spdiags([1/X_norms],[0],n,n))
	# X,Y,classes = load_stratified_covertype(strat_frac = strat_frac, sparse=sparse)
	
	# Test for stratified sampling
	# counts0 = {c:(Y0==c).sum() for c in classes}
	# counts = {c:(Y==c).sum() for c in classes}
	# s0 = sum(counts0.values())
	# s = sum(counts.values())

	# frac0 = {c:counts0[c]/s0 for c in classes}
	# frac = {c:counts[c]/s for c in classes}
	
	# import IPython
	# IPython.embed()


	import IPython
	IPython.embed()

	W0 = np.eye(d)
	
	init_pt = Y.nonzero()[0][nr.choice(len(Y.nonzero()[0]),2,replace=False)]

	prms = ASI.Parameters(pi=pi,sparse=sparse, verbose=True, eta=eta)
	slprms = SL.SPSDParameters(alpha=sl_alpha, C=sl_C, gamma=sl_gamma, margin=sl_margin, sampleR=sl_sampleR)

	kAS = ASI.kernelAS (prms)
	aAS = AAS.adaptiveKernelAS(W0, T, prms, slprms)

	# kAS.initialize(X_normalized,init_labels={init_pt:1})
	aAS.initialize(X_normalized,init_labels={p:1 for p in init_pt})

	hits1 = [1]
	hits2 = [1]

	for i in xrange(K):

		# idx1 = kAS.getNextMessage()
		idx2 = aAS.getNextMessage()

		# kAS.setLabelCurrent(Y[idx1])
		aAS.setLabelCurrent(Y[idx2])

		# hits1.append(hits1[-1]+Y[idx1])
		hits2.append(hits2[-1]+Y[idx2])

	import IPython
	IPython.embed()


if __name__ == '__main__':
	test_covtype()