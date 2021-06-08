from __future__ import division
from common import *
import random as rnd
import networkx as nx
import community as louvain
import igraph
from collections import defaultdict
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
	#print(max(list(G.nodes)))
	G2.add_vertices(max(list(G.nodes))+1)
	edge_list = []
	for edge in G.edges():
		edge_list.append((ShuffNodeMap[edge[0]], ShuffNodeMap[edge[1]]))
	G2.add_edges(edge_list)
	#print G2
	return G2

def getCommunitiesLouvain_old(G):
	#partition = louvain.best_partition(G,randomize=False)
	partition = louvain.best_partition(G)
	comm_list = []
	com_node_dic = defaultdict(list)
	for nodes in partition.keys():
		com_node_dic[partition[nodes]].append(nodes) #community:[nodes]
	for com in com_node_dic.keys():
		comm_list.append(com_node_dic[com])
	return comm_list

def getCommunitiesLouvain(G):
	#partition = louvain.best_partition(G,randomize=False)
	partition = louvain.best_partition(G)
	comm_list = []
	max_part = max(partition.values()) + 1#since partition starts from 0
	for com in range(max_part):
		#print(com)
		list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
		comm_list.append(list_nodes)
	return comm_list

def getCommunitiesLouvain_old2(G):
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
		temp_list = []
		for i in com:
			temp_list.append(i)
		comm_list.append(temp_list)
	return comm_list

def getLabelProp_old(G):
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
			for j in range(i, len(comm_set)):
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

def getBelongingness(G, algo):
	#print('===============================')
	if algo == 'louvain':#Louvain
		#print('Louvain')
		algoNo = 0
	elif algo == 'infomap':#Infomap
		#print('Infomap')
		algoNo = 1
	elif algo == 'lpm':#label propagation
		#print('Label Propagation')
		algoNo = 2
	else:#walk trap
		#print('WalkTrap')
		algoNo = 3
	#print('===============================')
	nodes = list(G.nodes())
	shuffNodes = list(G.nodes())
	edge_dic = {}
	#initialize the dictionary
	for edge in G.edges():
		if edge[0] < edge[1]:
			edge_dic[(edge[0], edge[1])] = 0
		else:
			edge_dic[(edge[1], edge[0])] = 0
	#perform the frequency computation
	for perm in range(no_of_perm):
		rnd.shuffle(shuffNodes) #randomly shuffle the graph
		ShuffNodeMap = initializeMap(nodes, shuffNodes) #forward map
		RevShuffNodeMap = initializeMap(shuffNodes, nodes) #reverse map
		#get the communities
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
		#compute the frequency of each edges
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


