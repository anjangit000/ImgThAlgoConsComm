import champ_functions
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import community as louvain
import infomap
import random as rnd
from tqdm import tqdm
from NMI import *
import copy as copy
from argparse import ArgumentParser


def getEdge(e):
	if e[0] < e[1]:
		edge = e
	else:
		edge = (e[1], e[0])
	return edge

def getCommunitiesLouvain(G, edge_dic):
	partition =  louvain.best_partition(G)
	for edge in G.edges():
		n1 = edge[0]
		n2 = edge[1]
		if n1>n2:
			n1, n2 = n2, n1
		if partition[n1] == partition[n2]:
			edge_dic[(n1, n2)] = edge_dic[(n1, n2)] + 1

	return [partition[node] for node in partition]


def getCommunitiesInfoMap(G, edge_dic):
	info = infomap.Infomap('--silent -s '+str(rnd.randint(0, 999999)))
	for e in list(G.edges()):
		    info.addLink(*e)
	info.run()
	partition = info.getModules() #node:community

	for edge in G.edges():
		n1 = edge[0]
		n2 = edge[1]
		if n1>n2:
			n1, n2 = n2, n1
		if partition[n1] == partition[n2]:
			edge_dic[(n1, n2)] = edge_dic[(n1, n2)] + 1

	return [partition[node] for node in partition]
	

def getLabelProp(G, edge_dic):
	p = nx.algorithms.community.asyn_lpa_communities(G)
	i = 0
	partition = {}
	for com in p:
		for c in com:
			partition[c] = i
		i += 1

	for edge in G.edges():
		n1 = edge[0]
		n2 = edge[1]
		if n1>n2:
			n1, n2 = n2, n1
		if partition[n1] == partition[n2]:
			edge_dic[(n1, n2)] = edge_dic[(n1, n2)] + 1

	return [partition[node] for node in partition]


def computeCM2(B, k, CM, edge_dic):
	for i in B:
		i = getEdge(i)
		if edge_dic[i] < no_of_perm:
			CM[k][0] += 1
		else:
			CM[k][1] += 1


def ComputePRA2(CM):
	if(CM[0][0] == 0):
		p0 = 0
		r0 = 0
	else:
		p0 = CM[0][0]/(CM[0][0]+CM[0][1])
		r0 = CM[0][0]/(CM[0][0]+CM[1][0])

	if(CM[1][1] == 0):
		p100 = 0
		r100 = 0
	else:
		p100 = CM[1][1]/(CM[1][0]+CM[1][1])
		r100 = CM[1][1]/(CM[0][1]+CM[1][1])

	if (p0+r0) == 0:
		F1_0 = 0
	else:
		F1_0 = 2*p0*r0/(p0+r0)

	if (p100+r100) == 0:
		F1_100 = 0
	else:
		F1_100 = 2*p100*r100/(p100+r100)

	print('Confusion Matrix: ',CM)
	print('*********************************************')
	print('Precision for 0: ',p0)
	print('Precision for 100: ',p100)
	print('*********************************************')
	print('Recall for 0: ',r0)
	print('Recall for 100: ',r100)
	print('*********************************************')
	print('Accuracy: ',(CM[0][0] + CM[1][1])/((CM[0][0]+CM[0][1]) + (CM[1][0]+CM[1][1])))
	print('*********************************************')
	print('F1-score for 0: ', F1_0)
	print('F1-score for 100: ', F1_100)
	print('*********************************************')
	print('Macro precision: ',(p0+p100)/2)
	print('Macro recall: ',(r0+r100)/2)
	print('*********************************************')
	print('Macro F1: ',(F1_0+F1_100)/2)

	return F1_100

def genGroundTruth(file1, file2):

	for e in G.edges():
		e0 = e[0]
		e1 = e[1]
		if(e0 > e1):
			e0, e1 = e1, e0

		if edge_dic[(e0,e1)] < no_of_perm:
			file1.write(str(e0)+' '+str(e1)+'\n')
		else:
			file2.write(str(e0)+' '+str(e1)+'\n')
no_of_perm = 100	
def get100EdgeList(edge_dic):
	EdgeList = []
	for edge in edge_dic:
		if edge_dic[edge] == no_of_perm:
			EdgeList.append(edge)
	return EdgeList

def connectedComponents(G, edgeList):
	tempG = nx.Graph()
	for edge in edgeList:
		tempG.add_edge(edge[0], edge[1])
	comlist = []
	for c in nx.connected_components(tempG):
		comlist.append(list(c))
	return comlist

def getCommunityFromEdgeDic(G, edge_dic):
	EdgeList = get100EdgeList(edge_dic)
	return connectedComponents(G, EdgeList)

def getChangedGT(edge_dic, max_val):
	temp_gt = copy.copy(edge_dic)
	for k in temp_gt:
		if temp_gt[k] >= max_val:
			edge_dic[k] = no_of_perm
	return edge_dic


if __name__ == '__main__':
	parser = ArgumentParser(description='Process some parameters.')
	parser.add_argument('-f', metavar='filename', type=str, nargs = '?', help='file with edgelist')
	parser.add_argument('-alg', metavar='alg', type=str, nargs = '?', default = 'louvain' , help='choose from \'louvain\' , \'infomap\' ,\'lpm\' ')
	args = parser.parse_args()

	G = nx.read_edgelist(args.f, nodetype=int)
	n_p = no_of_perm
	edge_dic = {}
	for edge in G.edges():
		if edge[0] < edge[1]:
			edge_dic[(edge[0], edge[1])] = 0
		else:
			edge_dic[(edge[1], edge[0])] = 0

	all_partition = []
	for i in range(n_p):
		if args.alg == 'louvain':
			partition = getCommunitiesLouvain(G, edge_dic)
		elif args.alg == 'infomap':
			partition = getCommunitiesInfoMap(G, edge_dic)
		elif args.alg == 'lpm':
			partition = getLabelProp(G, edge_dic)
		else:
			print('Choose proper algorithms')
			exit(0)
		all_partition.append(partition)

	'''
	edge_dic = getChangedGT(edge_dic, 80)
	file1 = open("grid_lou_gt0.txt","w") #ground truth for edge labelled 0
	file2 = open("grid_lou_gt100.txt", "w") #ground truth for edge labelled 100
	genGroundTruth(file1, file2)
	'''
	A_mat = nx.to_numpy_matrix(G)
	deg_list = [tup[1] for tup in G.degree()]
	P_mat = np.outer(deg_list, deg_list)

	coeff_array=champ_functions.create_coefarray_from_partitions(A_mat=A_mat, P_mat=P_mat, partition_array=np.array(all_partition))
	ind2doms=champ_functions.get_intersection(coef_array=coeff_array)

	gt_comm = getCommunityFromEdgeDic(G, edge_dic)
	f1_100 = 0
	nmi = 0
	for key in ind2doms:
		node_com_dic = {}
		B0 = []
		B100 = []

		for node, com in zip(G.nodes(), all_partition[key]):
			node_com_dic[node] =  com

		for e in G.edges():
			if node_com_dic[e[0]] == node_com_dic[e[1]]:
				B100.append(e)
			else:
				B0.append(e)

		CM = [[0,0],[0,0]]
		computeCM2(B0, 0, CM, edge_dic)
		computeCM2(B100, 1, CM, edge_dic)

		B100_comm = connectedComponents(G, B100)
		nmi += findNMI(B100_comm, gt_comm, G)
		f1_100 += ComputePRA2(CM)
		print("---------------------------------------------------------------------------")
	print("---------------------------------------------------------------------------")
	print("---------------------------------------------------------------------------")

	print('Avg F1 score for B100:',f1_100/len(ind2doms))
	print('nmi: ',nmi/len(ind2doms))

