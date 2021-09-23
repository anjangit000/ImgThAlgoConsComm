import networkx as nx
import numpy as np
import itertools
import sys
import os
import random
import igraph as ig
from networkx.algorithms import community
import networkx.algorithms.isolate
import community.community_louvain as cm
import math
import random
import argparse
import multiprocessing as mp
import matplotlib.pyplot as plt
import pickle as pkl
from util import *
from common import *
from belong import *
from NMI import *
import copy as copy

########################### fetching ground truth- start ###########################

def getGroundTruth(fileName1, fileName2):
	file100 = open(fileName2, "r")
	file0 = open(fileName1, "r")
	edge_dic = {}
	for l in file0:
		line = l.split(' ')
		u = int(line[0])
		v = int(line[1])
		edge_dic[(u,v)] = 0

	for l in file100:
		line = l.split(' ')
		u = int(line[0])
		v = int(line[1])
		edge_dic[(u,v)] = no_of_perm

	return edge_dic

########################### fetching ground truth- end ###########################

def check_consensus_graph(G, n_p, delta):
    '''
    This function checks if the networkx graph has converged.
    Input:
    G: networkx graph
    n_p: number of partitions while creating G
    delta: if more than delta fraction of the edges have weight != n_p then returns False, else True
    '''



    count = 0

    for wt in nx.get_edge_attributes(G, 'weight').values():
        if wt != 0 and wt != n_p:
            count += 1

    if count > delta*G.number_of_edges():
        return False

    return True



def nx_to_igraph(Gnx):
    '''
    Function takes in a network Graph, Gnx and returns the equivalent
    igraph graph g
    '''
    g = ig.Graph()
    g.add_vertices(sorted(Gnx.nodes()))
    g.add_edges(sorted(Gnx.edges()))
    g.es["weight"] = 1.0
    for edge in Gnx.edges():
        g[edge[0], edge[1]] = Gnx[edge[0]][edge[1]]['weight']
    return g


def group_to_partition(partition):
    '''
    Takes in a partition, dictionary in the format {node: community_membership}
    Returns a nested list of communities [[comm1], [comm2], ...... [comm_n]]
    '''

    part_dict = {}

    for index, value in partition.items():

        if value in part_dict:
            part_dict[value].append(index)
        else:
            part_dict[value] = [index]


    return part_dict.values()

def check_arguments(args):

    if(args.d > 0.2):
        print('delta is too high. Allowed values are between 0.02 and 0.2')
        return False
    if(args.d < 0.02):
        print('delta is too low. Allowed values are between 0.02 and 0.2')
        return False
    if(args.alg not in ('louvain', 'lpm', 'cnm', 'infomap')):
        print('Incorrect algorithm entered. run with -h for help')
        return False
    if (args.t > 1 or args.t < 0):
        print('Incorrect tau. run with -h for help')
        return False

    return True


def louvain_community_detection(networkx_graph):
    """
    Do louvain community detection
    :param networkx_graph:
    :return:
    """
    return cm.partition_at_level(cm.generate_dendrogram(networkx_graph, randomize=True, weight='weight'), 0)

def get_yielded_graph(graph, times):
    """
    Creates an iterator containing the same graph object multiple times. Can be used for applying multiprocessing map
    """
    for _ in range(times):
        yield graph

def fast_consensus(G,  algorithm = 'louvain', n_p = 20, thresh = 0.2, delta = 0.02):
    graph = G.copy()
    L = G.number_of_edges()
    N = G.number_of_nodes()

    for u,v in graph.edges():
        graph[u][v]['weight'] = 1.0

    while(True):

        if (algorithm == 'louvain'):
            nextgraph = graph.copy()
            L = G.number_of_edges()
            for u,v in nextgraph.edges():
                nextgraph[u][v]['weight'] = 0.0

            with mp.Pool(processes=mp.cpu_count()) as pool:
                communities_all = pool.map(louvain_community_detection, get_yielded_graph(graph, n_p))
            #print('>>>', communities_all)
            #print('\n')
            for node,nbr in graph.edges():

                if (node,nbr) in graph.edges() or (nbr, node) in graph.edges():
                    #print(graph[node][nbr]['weight'])
                    if graph[node][nbr]['weight'] not in (0,n_p):
                        #print(graph[node][nbr]['weight'])
                        for i in range(n_p):
                            communities = communities_all[i]
                            if communities[node] == communities[nbr]:
                                nextgraph[node][nbr]['weight'] += 1

            remove_edges = []
            for u,v in nextgraph.edges():
                if nextgraph[u][v]['weight'] < thresh*n_p:
                    remove_edges.append((u, v))

            nextgraph.remove_edges_from(remove_edges)


            if check_consensus_graph(nextgraph, n_p = n_p, delta = delta):
                break




            for _ in range(L):

                node = np.random.choice(nextgraph.nodes())
                neighbors = [a[1] for a in nextgraph.edges(node)]

                if (len(neighbors) >= 2):
                    a, b = random.sample(set(neighbors), 2)

                    if not nextgraph.has_edge(a, b):
                        nextgraph.add_edge(a, b, weight = 0)

                        for i in range(n_p):
                            communities = communities_all[i]

                            if communities[a] == communities[b]:
                                nextgraph[a][b]['weight'] += 1

            for node in nx.isolates(nextgraph):
                    nbr, weight = sorted(graph[node].items(), key=lambda edge: edge[1]['weight'])[0]
                    nextgraph.add_edge(node, nbr, weight = weight['weight'])

            graph = nextgraph.copy()


            if check_consensus_graph(nextgraph, n_p = n_p, delta = delta):
                break

        elif (algorithm in ('infomap', 'lpm')):

            nextgraph = graph.copy()

            for u,v in nextgraph.edges():
                nextgraph[u][v]['weight'] = 0.0

            if algorithm == 'infomap':
                communities = [{frozenset(c) for c in nx_to_igraph(graph).community_infomap().as_cover()} for _ in range(n_p)]
            if algorithm == 'lpm':
                communities = [{frozenset(c) for c in nx_to_igraph(graph).community_label_propagation().as_cover()} for _ in range(n_p)]


            for node, nbr in graph.edges():

                for i in range(n_p):
                    for c in communities[i]:
                        if node in c and nbr in c:
                            if not nextgraph.has_edge(node,nbr):
                                nextgraph.add_edge(node, nbr, weight = 0)
                            nextgraph[node][nbr]['weight'] += 1



            remove_edges = []
            for u,v in nextgraph.edges():
                if nextgraph[u][v]['weight'] < thresh*n_p:
                    remove_edges.append((u, v))
            nextgraph.remove_edges_from(remove_edges)



            for _ in range(L):
                node = np.random.choice(nextgraph.nodes())
                neighbors = [a[1] for a in nextgraph.edges(node)]

                if (len(neighbors) >= 2):
                    a, b = random.sample(set(neighbors), 2)

                    if not nextgraph.has_edge(a, b):
                        nextgraph.add_edge(a, b, weight = 0)

                        for i in range(n_p):
                            if a in communities[i] and b in communities[i]:
                                nextgraph[a][b]['weight'] += 1


            graph = nextgraph.copy()

            if check_consensus_graph(nextgraph, n_p = n_p, delta = delta):
                break

        elif (algorithm == 'cnm'):

            nextgraph = graph.copy()

            for u,v in nextgraph.edges():
                nextgraph[u][v]['weight'] = 0.0

            communities = []
            mapping = []
            inv_map = []


            for _ in range(n_p):

                order = list(range(N))
                random.shuffle(order)
                maps = dict(zip(range(N), order))

                mapping.append(maps)
                inv_map.append({v: k for k, v in maps.items()})
                G_c = nx.relabel_nodes(graph, mapping = maps, copy = True)
                G_igraph = nx_to_igraph(G_c)

                communities.append(G_igraph.community_fastgreedy(weights = 'weight').as_clustering())


            for i in range(n_p):

                edge_list = [(mapping[i][j], mapping[i][k]) for j,k in graph.edges()]

                for node,nbr in edge_list:
                    a, b = inv_map[i][node], inv_map[i][nbr]

                    if graph[a][b] not in (0, n_p):
                        for c in communities[i]:
                            if node in c and nbr in c:
                                nextgraph[a][b]['weight'] += 1


            remove_edges = []
            for u,v in nextgraph.edges():
                if nextgraph[u][v]['weight'] < thresh*n_p:
                    remove_edges.append((u, v))

            nextgraph.remove_edges_from(remove_edges)


            for _ in range(L):
                node = np.random.choice(nextgraph.nodes())
                neighbors = [a[1] for a in nextgraph.edges(node)]

                if (len(neighbors) >= 2):
                    a, b = random.sample(set(neighbors), 2)
                    if not nextgraph.has_edge(a, b):
                        nextgraph.add_edge(a, b, weight = 0)

                        for i in range(n_p):
                            for c in communities[i]:
                                if mapping[i][a] in c and mapping[i][b] in c:

                                    nextgraph[a][b]['weight'] += 1

            if check_consensus_graph(nextgraph, n_p, delta):
                break

        else:
            break

    if (algorithm == 'louvain'):
        with mp.Pool(processes=mp.cpu_count()) as pool:
            communities_all = pool.map(louvain_community_detection, get_yielded_graph(graph, n_p))
        return communities_all
    if algorithm == 'infomap':
        return [{frozenset(c) for c in nx_to_igraph(graph).community_infomap().as_cover()} for _ in range(n_p)]
    if algorithm == 'lpm':
        return [{frozenset(c) for c in nx_to_igraph(graph).community_label_propagation().as_cover()} for _ in range(n_p)]
    if algorithm == 'cnm':

        communities = []
        mapping = []
        inv_map = []

        for _ in range(n_p):
            order = list(range(N))
            random.shuffle(order)
            maps = dict(zip(range(N), order))

            mapping.append(maps)
            inv_map.append({v: k for k, v in maps.items()})
            G_c = nx.relabel_nodes(graph, mapping = maps, copy = True)
            G_igraph = nx_to_igraph(G_c)

            communities.append(G_igraph.community_fastgreedy(weights = 'weight').as_clustering())

        return communities

def computeB0B100(G, predictedClass, edgeDic):
	b0 = []
	b100 = []
	for e in G.edges():
		if e[0] < e[1]:
			edge = e
		else:
			edge = (e[1], e[0])

		u = e[0]
		v = e[1]
		if predictedClass[u] == predictedClass[v]:
			b100.append(edge)
		else:
			b0.append(edge)
	return b0, b100

def getTempGraphFromEdgeList(edge_list):
	tempG = nx.Graph()
	for edge in edge_list:
		tempG.add_edge(edge[0], edge[1])

	return tempG

def getChangedGT(edge_dic, max_val):
	temp_gt = copy.copy(edge_dic)
	for k in temp_gt:
		if temp_gt[k] >= val:
			edge_dic[k] = no_of_perm
	return edge_dic

		
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('-f', metavar='filename', type=str, nargs = '?', help='file with edgelist')
    parser.add_argument('-np', metavar='n_p', type=int, nargs = '?', default=20, help='number of input partitions for the algorithm (Default value: 20)')
    parser.add_argument('-t', metavar='tau', type=float, nargs = '?', help='used for filtering weak edges')
    parser.add_argument('-d', metavar='del', type=float,  nargs = '?', default = 0.02, help='convergence parameter (default = 0.02). Converges when less than delta proportion of the edges are with wt = 1')
    parser.add_argument('--alg', metavar='alg', type=str, nargs = '?', default = 'louvain' , help='choose from \'louvain\' , \'cnm\' , \'lpm\' , \'infomap\' ')
    parser.add_argument('-nmi', metavar='type', type=str, nargs = '?', default = 'True', help='choose from \'True\' , \'False\' ')
    args = parser.parse_args()

    default_tau = {'louvain': 0.2, 'cnm': 0.7 ,'infomap': 0.6, 'lpm': 0.8}
    if (args.t == None):
        args.t = default_tau.get(args.alg, 0.2)

    if check_arguments(args) == False:

        quit()

    G = nx.read_edgelist(args.f, nodetype=int)
    #G = nx.karate_club_graph()
    G = nx.convert_node_labels_to_integers(G, label_attribute = 'name')

    pre0 = []
    pre100 = []
    rec0 = []
    rec100 = []
    ACC = []
    F1_0 = []
    F1_100 = []
    MP = []
    MR = []
    MF1 = []


    for itr in range(100):

        output = fast_consensus(G, algorithm = args.alg, n_p = args.np, thresh = args.t, delta = args.d)

        if not os.path.exists('out_partitions'):
            os.makedirs('out_partitions')


        if(args.alg == 'louvain'):
            for i in range(len(output)):
                output[i] = group_to_partition(output[i])


        i = 0
        for partition in output:
            i += 1
            with open('out_partitions/' + str(i) , 'w') as f:
                for community in partition:
                    for node in community:
                        print(G.nodes[node]['name'], end = '\t', file = f)
                    print(file = f)


        #----------------------------------------mycode-----------------------------------------#
        file1 = open('out_partitions/1', 'r')
        Lines = file1.readlines()
        comlist = []
        for line in Lines:
            mylist = []
            for l in line.strip().split('\t'):
                mylist.append(int(l))
            comlist.append(mylist)
        print('------------------------------------------------------------')

        ourAlgo = 0
        if args.alg == 'infomap':
            ourAlgo = 1
        elif args.alg == 'lpm':
            ourAlgo = 3

        edge_dic = getBelongingness(G, no_of_perm, ourAlgo)
        #edge_dic = getChangedGT(edge_dic, 40)
        #edge_dic = getGroundTruth("../../firstPaper/groundTruth/lfr2/lfr2_lou_gt0.txt", "../../firstPaper/groundTruth/lfr2/lfr2_lou_gt100.txt")
        #edge_dic = getGroundTruth("../../firstPaper/groundTruth/com-dblpGT_10/gt00.txt", "../../firstPaper/groundTruth/com-dblpGT_10/gt1000.txt")
        for u,v in G.edges():
            G[u][v]['weight'] = edge_dic[(u,v)]


        b0, b100, B0, B100, GT0, GT100 = computeB0B100Fortunato(G, comlist, edge_dic)
        p0, p100, r0, r100, acc, f1_0, f1_100, m_p, mr, mf1 = computePRF_GCN(b0, b100, B0, B100, edge_dic)

        pre0.append(p0)
        pre100.append(p100)
        rec0.append(r0)
        rec100.append(r100)
        ACC.append(acc)
        F1_0.append(f1_0)
        F1_100.append(f1_100)
        MP.append(m_p)
        MR.append(mr)
        MF1.append(mf1)

    print('Precision for 0: ',getAvg(pre0))
    print('Precision for 100: ',getAvg(pre100))
    print('*********************************************')
    print('Recall for 0: ',getAvg(rec0))
    print('Recall for 100: ',getAvg(rec100))
    print('*********************************************')
    print('Accuracy: ',getAvg(ACC))
    print('*********************************************')
    print('F1-score for 0: ', getAvg(F1_0))
    print('F1-score for 100: ', getAvg(F1_100))
    print('*********************************************')
    print('Macro precision: ',getAvg(MP))
    print('Macro recall: ',getAvg(MR))
    print('*********************************************')
    print('Macro F1: ',getAvg(MF1))

    if args.nmi == 'True':
        gt_comm = getCommunityFromEdgeDic(G, edge_dic)
        findNMI(comlist, gt_comm, G)
    
    
