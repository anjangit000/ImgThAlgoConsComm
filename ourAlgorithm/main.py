from argparse import ArgumentParser
import networkx as nx
from belong import *
from util import *
from NMI import *

if __name__ == "__main__":
    parser = ArgumentParser(description='Process some parameters.')
    parser.add_argument('-f', metavar='filename', type=str, nargs = '?', help='file with edgelist')
    parser.add_argument('-alg', metavar='alg', type=str, nargs = '?', default = 'louvain' , help='choose from \'louvain\' , \'infomap\' ,\'lpm\' ')
    parser.add_argument('-th', metavar='type', type=str, nargs = '?', default = 'binary', help='choose from \'binary\' , \'multi\', \'multiItr\', \'multiItr+\' ')
    parser.add_argument('-nmi', metavar='type', type=str, nargs = '?', default = 'False', help='choose from \'True\' , \'False\' ')

    args = parser.parse_args()
    G = nx.read_edgelist(args.f, nodetype=int)
    #get the ground truth
    edge_dic = getBelongingness(G, args.alg)
    #compute the histogram
    print('computing histograms')
    if args.th == 'binary':
        B0, B100= computeFilteredBinaryOtsu(G, edge_dic)
    elif args.th == 'multi':
        B0, B100 = computeFilteredHist_multiOtsu_best2(G, edge_dic)
    elif args.th == 'multiItr':
        B0, B100 = computeFilteredHist_multiOtsu_repeatedHeighest(G, edge_dic)
    elif args.th == 'multiItr+':
        B0, B100 = computeFilteredHist_multiOtsu_repeatedHeighest_addWedge(G, edge_dic)
    elif args.th == 'convex':
        B0, B100= computeFilteredConvex(G, edge_dic)
    elif args.th == 'bilevel':
        B0, B100= computeFilteredBalanced(G, edge_dic) 
    elif args.th == 'gcn':
        B0, B100= gcn_optuna(G, edge_dic)

