from __future__ import division
from common import *
import random as rnd
import networkx as nx
import community as louvain
import igraph
#from networkx.algorithms.community import LFR_benchmark_graph

########################### computing the belongingness - start ###########################
def getNXGraph(G, ShuffNodeMap):
	G1 = nx.Graph()
	G1.add_nodes_from(G)
	for edge in G.edges():
		G1.add_edge(ShuffNodeMap[edge[0]], ShuffNodeMap[edge[1]])
	return G1

def getIGGraph(G, ShuffNodeMap):
	G2 = igraph.Graph()
	G2.add_vertices(G.number_of_nodes())
	edge_list = []
	for edge in G.edges():
		edge_list.append((ShuffNodeMap[edge[0]], ShuffNodeMap[edge[1]]))
	G2.add_edges(edge_list)
	#print G2
	return G2

def getCommunitiesLouvain(G):
	#partition = louvain.best_partition(G,randomize=False)
	partition = louvain.best_partition(G)
	comm_list = []
	for com in set(partition.values()):
		list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
		comm_list.append(list_nodes)
	return comm_list

def getCommunitiesInfoMap(G2):
	p = G2.community_infomap()
	comm_list = []
	for com in p:
		comm_list.append(com)
	return comm_list

def getCommunitiesWalkTrap(G2):
	wtrap = G2.community_walktrap(steps = 4)
	clust=wtrap.as_clustering()
	comm_list = []
	for com in clust:
		comm_list.append(com)
	return comm_list

def getLabelProp(G):
	#partition = nx.algorithms.community.label_propagation_communities(G)
	partition = nx.algorithms.community.asyn_lpa_communities(G)
	comm_list = []
	for com in partition:
		comm_list.append(list(com))
	return comm_list
	
def getFrequency(G, RevShuffNodeMap, next_level_communities, nodes, edge_dic):
	for comm_set in next_level_communities:
		#comm_set = sorted(comm_set)
		#print comm_set
		for i in range(len(comm_set)):
			for j in range(i, len(comm_set)):#why?????????????????????????????????
				if G.has_edge(RevShuffNodeMap[comm_set[i]], RevShuffNodeMap[comm_set[j]]):
					if RevShuffNodeMap[comm_set[i]] > RevShuffNodeMap[comm_set[j]]:
						edge_dic[(RevShuffNodeMap[comm_set[j]], RevShuffNodeMap[comm_set[i]])] = edge_dic[(RevShuffNodeMap[comm_set[j]], RevShuffNodeMap[comm_set[i]])] + 1
					else:
						edge_dic[(RevShuffNodeMap[comm_set[i]], RevShuffNodeMap[comm_set[j]])] = edge_dic[(RevShuffNodeMap[comm_set[i]], RevShuffNodeMap[comm_set[j]])] + 1

def initializeMap(nodes1, nodes2):
	nodemap = {}
	for m,n in zip(nodes1, nodes2):
		nodemap[m] = n
	return nodemap

def getBelongingness(G, no_of_perm, algoNo):
	print('===============================')
	if algoNo == 0:
		print('Louvain')
	elif algoNo == 1:
		print('Infomap')
	elif algoNo == 2:
		print('WalkTrap')
	else:
		print('Label Propagation')
	print('===============================')
	nodes = list(G.nodes())
	shuffNodes = list(G.nodes())
	edge_dic = {}
	for edge in G.edges():
		if edge[0] < edge[1]:
			edge_dic[(edge[0], edge[1])] = 0
		else:
			edge_dic[(edge[1], edge[0])] = 0
	for perm in range(no_of_perm):
		rnd.shuffle(shuffNodes)
		ShuffNodeMap = initializeMap(nodes, shuffNodes)
		RevShuffNodeMap = initializeMap(shuffNodes, nodes)
		if algoNo == 0:
			G1 = getNXGraph(G, ShuffNodeMap)
			next_level_communities = getCommunitiesLouvain(G1)
		elif algoNo == 1:
			G1 = getIGGraph(G, ShuffNodeMap)
			next_level_communities = getCommunitiesInfoMap(G1)
		elif algoNo == 2:
			G1 = getIGGraph(G, ShuffNodeMap)
			next_level_communities = getCommunitiesWalkTrap(G1)
		elif algoNo == 3:
			G1 = getNXGraph(G, ShuffNodeMap)
			next_level_communities = getLabelProp(G1)
		getFrequency(G, RevShuffNodeMap, next_level_communities, nodes, edge_dic)
	return edge_dic

def getMixedBelongingness(G, no_of_perm):
	print('===============================')
	print('Mixed community belongingness')
	print('===============================')
	nodes = list(G.nodes())
	shuffNodes = list(G.nodes())
	edge_dic = {}
	for edge in G.edges():
		edge_dic[(edge[0], edge[1])] = 0
	for perm in range(no_of_perm):
		rnd.shuffle(shuffNodes)
		ShuffNodeMap = initializeMap(nodes, shuffNodes)
		RevShuffNodeMap = initializeMap(shuffNodes, nodes)
		if perm%4 == 0:
			G1 = getNXGraph(G, ShuffNodeMap)
			next_level_communities = getCommunitiesLouvain(G1)
		elif perm%4 == 1:
			G1 = getIGGraph(G, ShuffNodeMap)
			next_level_communities = getCommunitiesInfoMap(G1)
		elif perm%4 == 2:
			G1 = getIGGraph(G, ShuffNodeMap)
			next_level_communities = getCommunitiesWalkTrap(G1)
		elif perm%4 == 3:
			G1 = getNXGraph(G, ShuffNodeMap)
			next_level_communities = getLabelProp(G1)
			
		getFrequency(G, RevShuffNodeMap, next_level_communities, nodes, edge_dic)
	return edge_dic

########################### computing the belongingness - end ###########################

###################program to find the constant community - start####################

def findConstantCommunity(G, itr_no):
	n = G.number_of_nodes()
	Vertex = [[-1 for x in range(itr_no)] for y in range(n)] 

	i = 0
	G_edges = G.edges()

	for perm in range(itr_no):
		nodes = list(G.nodes())
		rnd.shuffle(nodes)
		G1 = getNXGraph(G, nodes)
		next_level_communities = getCommunitiesLouvain(G1)# it is a list of sets.
		#if vertex v is in community then 
		for comm_set in next_level_communities:
			for v in comm_set:
				Vertex[nodes.index(v)][i] = next_level_communities.index(comm_set)
		i += 1
		G1.clear()

	j = 0

	constant_comm = {}#creating a dictionary 
	CC = []
	for v in range(0,n):
		temp = []
		if v not in constant_comm:# if v is not in any constant community
			constant_comm[v] = j	#insert v in a constant community
			temp.append(v)		
			for u in range(0,n):
				if constant_comm.get(u) != j:#for all u in V\CCj
					in_same_comm = True
					for i in range(0, itr_no):
						if Vertex[v][i] != Vertex[u][i]:
							in_same_comm = False
							break
					if in_same_comm == True:
						constant_comm[u] = j
						temp.append(u)
		if len(temp) > 1:
			CC.append(temp)
		j += 1

	'''
	#process constant_comm
	CC1 = []
	for i in range(len(CC)):
		if len(CC[i]) > 1:
			CC1.append(set(CC[i]))
	'''
	#print constant comm
	#print('constant communities: ', CC)
	return CC
	
###################program to find the constant community - end####################


