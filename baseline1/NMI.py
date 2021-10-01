from __future__ import division
import networkx as nx
import numpy as np
import math
from intbitset import intbitset
	
def findNMI(X,Y,G):
	p = np.zeros((len(X), len(Y)))
	p_ =  np.zeros(len(X))
	q_ = np.zeros(len(Y))
	#compute pij
	for i in range(len(X)):
		for j in range(len(Y)): 
			nij = len(intbitset(X[i]) & intbitset(Y[j]))#len(set(X[i]).intersection(set(Y[j]))) 
			#print(nij)
			p[i][j] = nij
			if nij == len(X[i]):
				break
	n = getNode(G)
	p = p/n
	#compute pi+
	for i in range(len(X)):
		for j in range(len(Y)): 
			p_[i] += p[i][j]

	#compute p+j
	for j in range(len(Y)):
		for i in range(len(X)):
			q_[j] += p[i][j]

	#compute I(X,Y)
	Ixy = 0
	for i in range(len(X)):
		for j in range(len(Y)):
			if p[i][j] == 0 or p_[i] == 0 or q_[j] == 0:
				continue
			Ixy += p[i][j]*(math.log(p[i][j]/(p_[i]*q_[j])))

	#compute H(X)
	Hx = 0
	for i in range(len(X)):
		if p_[i] != 0:
			Hx += p_[i]*(math.log(p_[i]))
	Hx = -Hx

	#computr H(Y):
	Hy = 0
	for j in range(len(Y)):
		if q_[j] != 0:
			Hy += q_[j]*(math.log(q_[j]))
	Hy = -Hy

	#compute NMI(X,Y)
	NMI = 2*Ixy/(Hx+Hy)

	print('NMI: ',NMI)
	return NMI

def get_int_deg(v, partition):
	nbd_v = G.neighbors(v)
	int_nbd_v = set(nbd_v).intersection(set(partition))
	return len(int_nbd_v)

def getSumofWeights(set_x):
	w = 0
	for node in set_x:
		w += get_int_deg(node, set_x)
	return w	 

def findModifiedNMI(X,Y):
	p = np.zeros((len(X), len(Y)))
	p_ =  np.zeros(len(X))
	q_ = np.zeros(len(Y))

	wplus = 0

	#compute wplus
	for i in range(n):
		wplus += G.degree[i]

	#compute p1ij
	for i in range(len(X)):
		for j in range(len(Y)): 
			xyij = set(X[i]).intersection(set(Y[j]))
			p[i][j] = getSumofWeights(xyij)

	p = p/wplus

	#compute pi+
	for i in range(len(X)):
		for j in range(len(Y)): 
			p_[i] += p[i][j]

	#compute p+j
	for j in range(len(Y)):
		for i in range(len(X)):
			q_[j] += p[i][j]

	#compute I(X,Y)
	Ixy = 0
	for i in range(len(X)):
		for j in range(len(Y)):
			if p[i][j] == 0 or p_[i] == 0 or q_[j] == 0:
				continue
			Ixy += p[i][j]*(math.log(p[i][j]/(p_[i]*q_[j])))

	#compute H(X)
	Hx = 0
	for i in range(len(X)):
		if p_[i] != 0:
			Hx += p_[i]*(math.log(p_[i]))
	Hx = -Hx

	#computr H(Y):
	Hy = 0
	for j in range(len(Y)):
		if q_[j] != 0:
			Hy += q_[j]*(math.log(q_[j]))
	Hy = -Hy

	#compute NMI(X,Y)
	M_NMI = 2*Ixy/(Hx+Hy)

	print('MNMI: ',M_NMI)

def findSymmDiff(X_alg,Y_ji):
	X = set([])
	Y = set([])

	for x in X_alg:
		for e in x:
			X.add(e)

	for y in Y_ji:
		for e in y:
			Y.add(e)

	if X.issubset(Y):
		print('alg is a subset of ji')
	if Y.issubset(X):
		print('ji is a subset of alg')
	sd = X.symmetric_difference(Y)
	print(len(sd))

#G=nx.read_edgelist("../TINY_100K/email_nx.txt", nodetype=int)
#n = 2000000
def getNode(G):
	return G.number_of_nodes()*800
'''
if __name__ == '__main__':
	print('email_nx')
	X_alg = [[0, 9, 15, 16, 19, 21, 23, 27, 30, 31, 33], [1, 12], [2, 3, 4, 8, 10, 13, 14, 18, 20, 22], [5, 6, 7, 11, 17], [24, 28], [25, 26, 29, 32]]

	Y_ji = [[0, 33, 9, 15, 16, 19, 21, 23, 27, 30, 31], [1, 12], [2, 3, 4, 8, 10, 13, 14, 18, 20, 22], [17, 11, 5, 6, 7], [24, 28], [32, 25, 26, 29]]

	findNMI(X_alg,Y_ji)
	findModifiedNMI(X_alg,Y_ji)
	findSymmDiff(X_alg,Y_ji)
'''
