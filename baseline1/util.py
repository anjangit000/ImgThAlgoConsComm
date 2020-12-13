from __future__ import division
from common import *
import operator
import networkx as nx
import pickle
from NMI import *

########################### Precision,Recall and Accuracy - start ###########################

def computeCM(B, k, CM):
	for i in B:
		if i == 0:
			CM[k][0] += 1
		elif i > 0 and i < no_of_perm:
			CM[k][1] += 1
		else:
			CM[k][2] += 1

def computeCM2(B, L, k, CM, edge_dic):
	col0 = []
	col1 = []
	for i,j in zip(B,L):
		if edge_dic[i] < no_of_perm:
			CM[k][0] += 1
			col0.append(j)
		else:
			CM[k][1] += 1
			col1.append(j)
	return col0, col1

def ComputePRA(CM):
	if(CM[0][0] == 0):
		p0 = 0
		r0 = 0
	else:
		p0 = CM[0][0]/(CM[0][0]+CM[0][1]+CM[0][2])
		r0 = CM[0][0]/(CM[0][0]+CM[1][0]+CM[2][0])

	if(CM[1][1] == 0):
		pM = 0
		rM = 0
	else:
		pM = CM[1][1]/(CM[1][0]+CM[1][1]+CM[1][2])
		rM = CM[1][1]/(CM[0][1]+CM[1][1]+CM[2][1])

	if(CM[2][2] == 0):
		p100 = 0
		r100 = 0
	else:
		p100 = CM[2][2]/(CM[2][0]+CM[2][1]+CM[2][2])
		r100 = CM[2][2]/(CM[0][2]+CM[1][2]+CM[2][2])

	if (p0+r0) == 0:
		F1_0 = 0
	else:
		F1_0 = 2*p0*r0/(p0+r0)

	if (pM+rM) == 0:
		F1_M = 0
	else:
		F1_M = 2*pM*rM/(pM+rM)
	
	if (p100+r100) == 0:
		F1_100 = 0
	else:
		F1_100 = 2*p100*r100/(p100+r100)
	
	print('Confusion Matrix: ',CM)
	print('*********************************************')
	print('Precision for 0: ',p0)
	print('Precision for Middle: ',pM)
	print('Precision for 100: ',p100)
	print('*********************************************')
	print('Recall for 0: ',r0)
	print('Recall for Middle: ',rM)
	print('Recall for 100: ',r100)
	print('*********************************************')
	print('Accuracy: ',(CM[0][0] + CM[1][1] + CM[2][2])/((CM[0][0]+CM[0][1]+CM[0][2]) + (CM[1][0]+CM[1][1]+CM[1][2]) + (CM[2][0]+CM[2][1]+CM[2][2])))
	print('*********************************************')
	print('F1-score for 0: ', F1_0)
	print('F1-score for Middle: ', F1_M)
	print('F1-score for 100: ', F1_100)
	print('*********************************************')
	print('Macro precision: ',(p0+pM+p100)/3)
	print('Macro recall: ',(r0+rM+r100)/3)
	print('*********************************************')
	print('Macro F1: ',(F1_0+F1_M+F1_100)/3)

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

	acc = (CM[0][0] + CM[1][1])/((CM[0][0]+CM[0][1]) + (CM[1][0]+CM[1][1]))
	'''
	print('Confusion Matrix: ',CM)
	print('*********************************************')
	print('Precision for 0: ',p0)
	print('Precision for 100: ',p100)
	print('*********************************************')
	print('Recall for 0: ',r0)
	print('Recall for 100: ',r100)
	print('*********************************************')
	print('Accuracy: ',a)
	print('*********************************************')
	print('F1-score for 0: ', F1_0)
	print('F1-score for 100: ', F1_100)
	print('*********************************************')
	print('Macro precision: ',(p0+p100)/2)
	print('Macro recall: ',(r0+r100)/2)
	print('*********************************************')
	print('Macro F1: ',(F1_0+F1_100)/2)
	'''
	
	return p0, p100, r0, r100, acc, F1_0, F1_100, (p0+p100)/2, (r0+r100)/2, (F1_0+F1_100)/2
########################### Precision,Recall and Accuracy - end ###########################

########################### computing the parameters - start ###########################
def getEdgeCount(G, nodeSet):
	edgeCount = 0
	for x in nodeSet:
		for y in nodeSet:			
			if G.has_edge(x,y):
				edgeCount += 1
	return edgeCount

def getEdgeCount2(G, nodeSet1, nodeSet2):
	edgeCount = 0
	for x in nodeSet1:
		for y in nodeSet2:			
			if G.has_edge(x,y):
				edgeCount += 1
	return edgeCount

def cluscoeff(G, nodeSet):
	outdegree = len(nodeSet)
	num = getEdgeCount(G, nodeSet)
	deno = outdegree*(outdegree-1)
	if outdegree == 0 or outdegree == 1:
		return 0
	return num/deno

def JI(nbd_v1, nbd_v2):#This is jaccard index
	set_union = nbd_v1.union(nbd_v2)
	set_intersection = nbd_v1.intersection(nbd_v2)
	if len(set_union) == 0:
		return 0
	return len(set_intersection)/len(set_union)

#method to compute a JI for a given edge
def computeJI(G, edge):
	v1 = edge[0]
	v2 = edge[1]
	nbd_v1 = set(G.neighbors(v1))
	nbd_v2 = set(G.neighbors(v2))	
	return JI(nbd_v1, nbd_v2)

def computeCC(G, edge):
	v1 = edge[0]
	v2 = edge[1]
	nbd_v1 = set(G.neighbors(v1))
	nbd_v2 = set(G.neighbors(v2))
	nodeSet = nbd_v1.intersection(nbd_v2)
	#nodeSet.add(v1)
	#nodeSet.add(v2)
	edgeCount = 0
	for x in nodeSet:
		for y in nodeSet:			
			if G.has_edge(x,y):
				edgeCount += 1
	outdegree = len(nodeSet)
	num = edgeCount
	deno = outdegree*(outdegree-1)
	if outdegree == 0 or outdegree == 1:
		return 0
	return num/deno

def computeModifiedCC(G, edge):
	v1 = edge[0]
	v2 = edge[1]
	nbd_v1 = set(G.neighbors(v1))
	nbd_v2 = set(G.neighbors(v2))
	nodeSet = nbd_v1.union(nbd_v2)
	#nodeSet = nodeSet.difference(set((v1,v2)))
	edgeCount = 0
	for x in nodeSet:
		for y in nodeSet:			
			if G.has_edge(x,y):
				edgeCount += 1
	outdegree = len(nodeSet)
	num = edgeCount
	deno = outdegree*(outdegree-1)
	if outdegree == 0 or outdegree == 1:
		return 0
	return num/deno

#cc for symmetric difference
def computeUncommonCC(G, edge):
	v1 = edge[0]
	v2 = edge[1]
	nbd_v1 = set(G.neighbors(v1))
	nbd_v2 = set(G.neighbors(v2))
	nodeSet1 = nbd_v1.difference(nbd_v2)
	nodeSet2 = nbd_v2.difference(nbd_v1)
	nodeSet1.remove(v2)
	nodeSet2.remove(v1)
	edgeCount = 0
	for x in nodeSet1:
		for y in nodeSet2:			
			if G.has_edge(x,y):
				edgeCount += 1
	
	#outdegree = min(len(nodeSet1) , len(nodeSet2))
	outdegree = len(nodeSet1.union(nodeSet2))
	num = edgeCount
	deno = outdegree*(outdegree-1)
	if outdegree == 0 or outdegree == 1:
		return 0
	return num/deno

#cc for symmetric difference
def computeUncommonCC2(G, edge):
	v1 = edge[0]
	v2 = edge[1]
	nbd_v1 = set(G.neighbors(v1))
	nbd_v2 = set(G.neighbors(v2))
	nodeSet1 = nbd_v1.difference(nbd_v2)
	nodeSet2 = nbd_v2.difference(nbd_v1)
	nodeSet1.remove(v2)
	nodeSet2.remove(v1)
	edgeCount = 0
	for x in nodeSet1:
		for y in nodeSet2:			
			if G.has_edge(x,y):
				edgeCount += 1
	
	#outdegree = min(len(nodeSet1) , len(nodeSet2))
	outdegree = len(nodeSet1)*len(nodeSet2)
	num = edgeCount
	deno = outdegree
	if outdegree == 0 or outdegree == 1:
		return 0
	return num/deno

def ComputeOnlyNbd(G, edge):
	v1 = edge[0]
	v2 = edge[1]
	nbd_v1 = set(G.neighbors(v1))
	nbd_v2 = set(G.neighbors(v2))
	common = nbd_v1.intersection(nbd_v2)
	only_nbd_v1 = nbd_v1.difference(common)
	only_nbd_v2 = nbd_v2.difference(common)
	OLcc = cluscoeff(G, only_nbd_v1)
	ORcc = cluscoeff(G, only_nbd_v2)
	Lcc = cluscoeff(G, nbd_v1)
	Rcc = cluscoeff(G, nbd_v2)
	UMC = cluscoeff(G, only_nbd_v1.union(only_nbd_v2))
	OLE = getEdgeCount(G, only_nbd_v1)
	ORE = getEdgeCount(G, only_nbd_v2)
	LE = getEdgeCount(G, nbd_v1)
	RE = getEdgeCount(G, nbd_v2)
	CE = getEdgeCount(G, common)
	UE = getEdgeCount2(G, only_nbd_v1, only_nbd_v2)
	return OLcc, ORcc, UMC

def getEdgeNbd(G, edge):
	v1 = edge[0]
	v2 = edge[1]
	nbd_v1 = set(G.neighbors(v1))
	nbd_v2 = set(G.neighbors(v2))
	return len(nbd_v1.union(nbd_v2))


def computeParameters(G):

	JIlist = []
	CClist = []
	MccList = []
	UccList = []
	LccList = []
	RccList = []
	UMCList = []
	edgeNbdList = []

	for edge in G.edges():
		ji = computeJI(G, edge)
		cc = computeCC(G, edge)
		mcc = computeModifiedCC(G, edge)
		ucc = computeUncommonCC2(G, edge)
		Lcc, Rcc, UMC = ComputeOnlyNbd(G, edge)
		JIlist.append(ji)
		CClist.append(cc)
		MccList.append(mcc)
		UccList.append(ucc)
		LccList.append(Lcc)
		RccList.append(Rcc)
		UMCList.append(UMC)
		totalnbd = getEdgeNbd(G, edge)
		edgeNbdList.append((edge, totalnbd))

	return JIlist, CClist, MccList, LccList, UccList, RccList, UMCList, edgeNbdList

########################### computing the parameters - end ###########################

########################### computing the Filter class - start ###########################
def getLabelledNodes(G, edge_dic):
	constant_nodes = set([])
	for e in G.edges():#collect the nodes of a constant community
		if e[0] < e[1]:
			edge = e
		else:
			edge = (e[1], e[0])

		if edge_dic[edge] == no_of_perm:
			constant_nodes.add(edge[0])
			constant_nodes.add(edge[1])
	
	nodes = set(G.nodes())
	nonCons_nodes = nodes.difference(constant_nodes)

def computeFilteredHist(G, edge_dic, fileName):
	JIlist, CClist, MccList, LccList, UccList, RccList, UMCList, edgeNbdList = computeParameters(G)
	T_ji_otsu = computeOtsuThreshold(edgeNbdList, JIlist)
	T_mcc_otsu = computeOtsuThreshold(edgeNbdList, MccList)
	T_cc_otsu = computeOtsuThreshold(edgeNbdList, CClist)
	T_ucc_otsu = computeOtsuThreshold(edgeNbdList, UccList)
	T_lcc_otsu = computeOtsuThreshold(edgeNbdList, LccList)
	T_rcc_otsu = computeOtsuThreshold(edgeNbdList, RccList)
	T_umc_otsu = computeOtsuThreshold(edgeNbdList, UMCList)

	'''
	T_ji_km = computeThreshold(G, edgeNbdList, JIlist, 'ji')
	T_mcc_km = computeThreshold(G, edgeNbdList, MccList, 'mcc')
	T_cc_km = computeThreshold(G, edgeNbdList, CClist, 'cc')
	T_ucc_km = computeThreshold(G, edgeNbdList, UccList, 'ucc')
	T_lcc_km = computeThreshold(G, edgeNbdList, LccList, 'lcc')
	T_rcc_km = computeThreshold(G, edgeNbdList, RccList, 'rcc')
	T_umc_km = computeThreshold(G, edgeNbdList, UMCList, 'umc')

	
	T_ji_mean, T_cc_mean, T_mcc_mean, T_ucc_mean, T_Lcc_mean, T_Rcc_mean, T_umc_mean = computeMean([JIlist, CClist, MccList, UccList, LccList, RccList, UMCList])
	'''
	CM = [[0,0,0],[0,0,0],[0,0,0]]

	B0 = []
	B100 = []
	BM = []

	actual_count0 = 0
	actual_count100 = 0
	actual_countm = 0

	#print 'Otsu: ',T_ji_otsu, T_cc_otsu, T_mcc_otsu, T_ucc_otsu, T_lcc_otsu, T_rcc_otsu, T_umc_otsu
	#print 'Global Mean', T_ji_mean, T_cc_mean, T_mcc_mean, T_ucc_mean, T_Lcc_mean, T_Rcc_mean
	#print 'K-mean', T_ji_km, T_cc_km, T_mcc_km, T_ucc_km, T_lcc_km, T_rcc_km
	f = open(fileName, "a")

	for e in G.edges():
		if e[0] < e[1]:
			edge = e
		else:
			edge = (e[1], e[0])
		#Mnbdji, Mnbdcc, Mnbdmcc ,Mnbducc, Mnbdlcc, Mnbdrcc = computeMeanNbdProp(G,edge)#for adaptive threshold
		cc = computeCC(G, edge)
		mcc = computeModifiedCC(G, edge)
		ji = computeJI(G, edge)
		ucc = computeUncommonCC2(G, edge)
		Lcc, Rcc, UMC = ComputeOnlyNbd(G, edge)
		if (ji >= 0 and ji < T_ji_otsu/2 and cc >= 0 and cc < T_cc_otsu/2 and ucc >= 0 and ucc < T_ucc_otsu/2 and Rcc > T_lcc_otsu and Lcc > T_rcc_otsu):
			B0.append(edge_dic[edge])
		elif (cc > T_cc_otsu/2 and ji > T_ji_otsu ) or (cc == 0 and mcc > T_mcc_otsu and ucc > T_ucc_otsu):# or (cc == 0 and UMC > T_umc_otsu and ucc > T_ucc_otsu):
			B100.append(edge_dic[edge])
		else:
			BM.append(edge_dic[edge])

		if edge_dic[edge] == 0:
			#print edge, '+++'#cc, ji, mcc, ucc, Lcc, Rcc
			#f.write(str(edge)+','+ str(cc)+','+str(ji)+','+str(mcc)+','+str(ucc)+','+str(Lcc)+','+str(Rcc)+','+str(UMC)+','+str(edge_dic[edge])+'\n')

			actual_count0 += 1
		elif edge_dic[edge] == no_of_perm:
			#f.write(str(edge)+','+ str(cc)+','+str(ji)+','+str(mcc)+','+str(ucc)+','+str(Lcc)+','+str(Rcc)+','+str(UMC)+','+str(edge_dic[edge])+'\n')
			#if(cc == 0 and ucc < 0.1):
				#print edge, edge_dic[edge], cc, ji, mcc, ucc, Lcc, Rcc, '---'
			actual_count100 += 1
		else:
			actual_countm += 1


	f.close()	

	print('********************PREDICTED********************')
	
	print('Belongingness 0: ',len(B0))
	print('Belongingness middle: ',len(BM))
	print('Belongingness 100: ',len(B100))


	print('********************ACTUAL********************')
	print('Belongingness 0: ',actual_count0)
	print('Belongingness middle: ',actual_countm)
	print('Belongingness 100: ',actual_count100)

	print('*********************************************')
	computeCM(B0, 0, CM)
	computeCM(BM, 1, CM)
	computeCM(B100, 2, CM)
	ComputePRA(CM)



def computeTruthTable(G, edge_dic, fileName):
	JIlist, CClist, MccList, LccList, UccList, RccList, UMCList, edgeNbdList = computeParameters(G)
	T_ji_otsu = computeOtsuThreshold(edgeNbdList, JIlist)
	T_mcc_otsu = computeOtsuThreshold(edgeNbdList, MccList)
	T_cc_otsu = computeOtsuThreshold(edgeNbdList, CClist)
	T_ucc_otsu = computeOtsuThreshold(edgeNbdList, UccList)
	T_lcc_otsu = computeOtsuThreshold(edgeNbdList, LccList)
	T_rcc_otsu = computeOtsuThreshold(edgeNbdList, RccList)
	T_umc_otsu = computeOtsuThreshold(edgeNbdList, UMCList)

	edgeFreqList = []
	total = 43
	for i in range(total):
		edgeFreqList.append([])

	for edge in G.edges():
		cc = computeCC(G, edge)
		mcc = computeModifiedCC(G, edge)
		Lcc, Rcc, UMC = ComputeOnlyNbd(G, edge)
		if  mcc < T_mcc_otsu and cc < T_cc_otsu and Lcc < T_lcc_otsu and Rcc < T_rcc_otsu and UMC < T_umc_otsu:
			edgeFreqList[0].append(edge_dic[edge])
		if  mcc < T_mcc_otsu and cc < T_cc_otsu and Lcc < T_lcc_otsu and Rcc < T_rcc_otsu and UMC >= T_umc_otsu:
			edgeFreqList[1].append(edge_dic[edge])
		if  mcc < T_mcc_otsu and cc < T_cc_otsu and Lcc < T_lcc_otsu and Rcc >= T_rcc_otsu and UMC < T_umc_otsu:
			edgeFreqList[2].append(edge_dic[edge])
		if  mcc < T_mcc_otsu and cc < T_cc_otsu and Lcc < T_lcc_otsu and Rcc >= T_rcc_otsu and UMC >= T_umc_otsu:
			edgeFreqList[3].append(edge_dic[edge])
		if  mcc < T_mcc_otsu and cc < T_cc_otsu and Lcc >= T_lcc_otsu and Rcc < T_rcc_otsu and UMC < T_umc_otsu:
			edgeFreqList[4].append(edge_dic[edge])
		if  mcc < T_mcc_otsu and cc < T_cc_otsu and Lcc >= T_lcc_otsu and Rcc < T_rcc_otsu and UMC >= T_umc_otsu:
			edgeFreqList[5].append(edge_dic[edge])
		if  mcc < T_mcc_otsu and cc < T_cc_otsu and Lcc >= T_lcc_otsu and Rcc >= T_rcc_otsu and UMC < T_umc_otsu:
			edgeFreqList[6].append(edge_dic[edge])
		if  mcc < T_mcc_otsu and cc < T_cc_otsu and Lcc >= T_lcc_otsu and Rcc >= T_rcc_otsu and UMC >= T_umc_otsu:
			edgeFreqList[7].append(edge_dic[edge])
		if  mcc < T_mcc_otsu and cc >= T_cc_otsu and Lcc < T_lcc_otsu and Rcc < T_rcc_otsu and UMC < T_umc_otsu:
			edgeFreqList[8].append(edge_dic[edge])
		if  mcc < T_mcc_otsu and cc >= T_cc_otsu and Lcc < T_lcc_otsu and Rcc < T_rcc_otsu and UMC >= T_umc_otsu:
			edgeFreqList[9].append(edge_dic[edge])
		if  mcc < T_mcc_otsu and cc >= T_cc_otsu and Lcc < T_lcc_otsu and Rcc >= T_rcc_otsu and UMC < T_umc_otsu:
			edgeFreqList[10].append(edge_dic[edge])
		if  mcc < T_mcc_otsu and cc >= T_cc_otsu and Lcc < T_lcc_otsu and Rcc >= T_rcc_otsu and UMC >= T_umc_otsu:
			edgeFreqList[11].append(edge_dic[edge])
		if  mcc < T_mcc_otsu and cc >= T_cc_otsu and Lcc >= T_lcc_otsu and Rcc < T_rcc_otsu and UMC < T_umc_otsu:
			edgeFreqList[12].append(edge_dic[edge])
		if  mcc < T_mcc_otsu and cc >= T_cc_otsu and Lcc >= T_lcc_otsu and Rcc < T_rcc_otsu and UMC >= T_umc_otsu:
			edgeFreqList[13].append(edge_dic[edge])
		if  mcc < T_mcc_otsu and cc >= T_cc_otsu and Lcc >= T_lcc_otsu and Rcc >= T_rcc_otsu and UMC < T_umc_otsu:
			edgeFreqList[14].append(edge_dic[edge])
		if  mcc < T_mcc_otsu and cc >= T_cc_otsu and Lcc >= T_lcc_otsu and Rcc >= T_rcc_otsu and UMC >= T_umc_otsu:
			edgeFreqList[15].append(edge_dic[edge])
		if  mcc >= T_mcc_otsu and cc < T_cc_otsu and Lcc < T_lcc_otsu and Rcc < T_rcc_otsu and UMC < T_umc_otsu:
			edgeFreqList[16].append(edge_dic[edge])
		if  mcc >= T_mcc_otsu and cc < T_cc_otsu and Lcc < T_lcc_otsu and Rcc < T_rcc_otsu and UMC >= T_umc_otsu:
			edgeFreqList[17].append(edge_dic[edge])
		if  mcc >= T_mcc_otsu and cc < T_cc_otsu and Lcc < T_lcc_otsu and Rcc >= T_rcc_otsu and UMC < T_umc_otsu:
			edgeFreqList[18].append(edge_dic[edge])
		if  mcc >= T_mcc_otsu and cc < T_cc_otsu and Lcc < T_lcc_otsu and Rcc >= T_rcc_otsu and UMC >= T_umc_otsu:
			edgeFreqList[19].append(edge_dic[edge])
		if  mcc >= T_mcc_otsu and cc < T_cc_otsu and Lcc >= T_lcc_otsu and Rcc < T_rcc_otsu and UMC < T_umc_otsu:
			edgeFreqList[20].append(edge_dic[edge])
		if  mcc >= T_mcc_otsu and cc < T_cc_otsu and Lcc >= T_lcc_otsu and Rcc < T_rcc_otsu and UMC >= T_umc_otsu:
			edgeFreqList[21].append(edge_dic[edge])
		if  mcc >= T_mcc_otsu and cc < T_cc_otsu and Lcc >= T_lcc_otsu and Rcc >= T_rcc_otsu and UMC < T_umc_otsu:
			edgeFreqList[22].append(edge_dic[edge])
		if  mcc >= T_mcc_otsu and cc < T_cc_otsu and Lcc >= T_lcc_otsu and Rcc >= T_rcc_otsu and UMC >= T_umc_otsu:
			edgeFreqList[23].append(edge_dic[edge])
		if  mcc >= T_mcc_otsu and cc >= T_cc_otsu and Lcc < T_lcc_otsu and Rcc < T_rcc_otsu and UMC < T_umc_otsu:
			edgeFreqList[24].append(edge_dic[edge])
		if  mcc >= T_mcc_otsu and cc >= T_cc_otsu and Lcc < T_lcc_otsu and Rcc < T_rcc_otsu and UMC >= T_umc_otsu:
			edgeFreqList[25].append(edge_dic[edge])
		if  mcc >= T_mcc_otsu and cc >= T_cc_otsu and Lcc < T_lcc_otsu and Rcc >= T_rcc_otsu and UMC < T_umc_otsu:
			edgeFreqList[26].append(edge_dic[edge])
		if  mcc >= T_mcc_otsu and cc >= T_cc_otsu and Lcc < T_lcc_otsu and Rcc >= T_rcc_otsu and UMC >= T_umc_otsu:
			edgeFreqList[27].append(edge_dic[edge])
		if  mcc >= T_mcc_otsu and cc >= T_cc_otsu and Lcc >= T_lcc_otsu and Rcc < T_rcc_otsu and UMC < T_umc_otsu:
			edgeFreqList[28].append(edge_dic[edge])
		if  mcc >= T_mcc_otsu and cc >= T_cc_otsu and Lcc >= T_lcc_otsu and Rcc < T_rcc_otsu and UMC >= T_umc_otsu:
			edgeFreqList[29].append(edge_dic[edge])
		if  mcc >= T_mcc_otsu and cc >= T_cc_otsu and Lcc >= T_lcc_otsu and Rcc >= T_rcc_otsu and UMC < T_umc_otsu:
			edgeFreqList[30].append(edge_dic[edge])
		if  mcc >= T_mcc_otsu and cc >= T_cc_otsu and Lcc >= T_lcc_otsu and Rcc >= T_rcc_otsu and UMC >= T_umc_otsu:
			edgeFreqList[31].append(edge_dic[edge])

		if  mcc < T_mcc_otsu:
			edgeFreqList[32].append(edge_dic[edge])
		if  cc < T_cc_otsu:
			edgeFreqList[33].append(edge_dic[edge])
		if  Lcc < T_lcc_otsu:
			edgeFreqList[34].append(edge_dic[edge])
		if  Rcc < T_rcc_otsu:
			edgeFreqList[35].append(edge_dic[edge])
		if  UMC < T_umc_otsu:
			edgeFreqList[36].append(edge_dic[edge])

		if  mcc >= T_mcc_otsu:
			edgeFreqList[37].append(edge_dic[edge])
		if  cc >= T_cc_otsu:
			edgeFreqList[38].append(edge_dic[edge])
		if  Lcc >= T_lcc_otsu:
			edgeFreqList[39].append(edge_dic[edge])
		if  Rcc >= T_rcc_otsu:
			edgeFreqList[40].append(edge_dic[edge])
		if  UMC >= T_umc_otsu:
			edgeFreqList[41].append(edge_dic[edge])

		edgeFreqList[42].append(edge_dic[edge])

	f = open(fileName, "a")
	for i in range(total):
		efl = edgeFreqList[i]
		edgeFreqDic = {x:efl.count(x) for x in efl}
		#print edgeFreqDic
		f1 = f2 = f3 = f4 = f5 = 0
		for k in edgeFreqDic:
			if k >= 0 and k<=1000:
				f1 += edgeFreqDic[k]
			if k >= 1001 and k<=2000:
				f2 += edgeFreqDic[k]
			if k >= 2001 and k<=3000:
				f3 += edgeFreqDic[k]
			if k >= 3001 and k<=4000:
				f4 += edgeFreqDic[k]
			if k >= 4001 and k<=no_of_perm:
				f5 += edgeFreqDic[k]
		
		f.write(str(f1)+','+str(f2)+','+str(f3)+','+str(f4)+','+str(f5)+'\n')
	f.close()	

########################### computing the Filter class - end ###########################

########################### computing threshold - start ###########################
def getAvgcc(G, sortedEdgeNbdList, index, cctype):
	listLen = len(sortedEdgeNbdList)-1
	to = 10
	if index == 1:
		fewList =  sortedEdgeNbdList[index-1:to]
	else:
		fewList = sortedEdgeNbdList[listLen:listLen-to:-1]
	sumMcc = 0
	for t in fewList:
		Lcc, Rcc, UMC = ComputeOnlyNbd(G, t[0])
		if cctype == 'mcc':
			sumMcc = sumMcc + computeModifiedCC(G, t[0])
		elif cctype == 'cc':
			sumMcc = sumMcc + computeCC(G, t[0])
		elif cctype == 'ji':
			sumMcc = sumMcc + computeJI(G, t[0])
		elif cctype == 'ucc':
			sumMcc = sumMcc + computeUncommonCC(G, t[0])
		elif cctype == 'lcc':
			sumMcc = sumMcc + Lcc
		elif cctype == 'rcc':
			sumMcc = sumMcc + Rcc
		elif cctype == 'umc':
			sumMcc = sumMcc + UMC

	return sumMcc/10

def getAvgccBesidesT(T, ccList):
	left = 0
	right = 0
	left_m = 0
	right_m = 0
	for m in ccList:
		if m <= T:
			left += 1
			left_m += m
		else:
			right += 1
			right_m += m
	if left == 0 and right == 0:
		return 0,0
	elif left == 0:
		return 0, (right_m/right)
	elif right == 0:
		return (left_m/left), 0

	return (left_m/left), (right_m/right)

def computeThreshold(G, edgeNbdList, ccList, cctype):
	#find 10 edges having maximum neighbours and find their miuB(T)
	#find 10 edges having minimum neighbours and find their miuO(T)
	# T = miuB(T)+miuO(T)/2
	# find the miuB(T) and miuO(T) once again from left and right side of T
	# recalculate T. If new T is not a delta distance from the old T, break else go to the prev state
	sortedEdgeNbdList =  sorted(edgeNbdList, key=operator.itemgetter(1), reverse=True)
	miO_T = getAvgcc(G, sortedEdgeNbdList, 1, cctype)
	miB_T = getAvgcc(G, sortedEdgeNbdList, -1, cctype)
	count = 0
	while True:
		count += 1
		T =  (miO_T + miB_T)/2
		miO_T, miB_T = getAvgccBesidesT(T, ccList)
		if T != (miO_T + miB_T)/2:
			T = (miO_T + miB_T)/2
		elif count == 100:
			break
		else:
			break
	return T

def getPixelCountsAndAvg(T, ccList):
	left = 0
	right = 0
	left_m = 0
	right_m = 0
	ctr = collections.Counter(ccList)
	for c in ctr:
		if c <= T:
			left += ctr[c]
			left_m += c*ctr[c]
		else:
			right += ctr[c]
			right_m += c*ctr[c]
			
	if left == 0 and right == 0:
		return 0,0,0,0
	elif left == 0:
		return 0, right, 0, (right_m/right)
	elif right == 0:
		return left, 0, (left_m/left), 0
	return left, right, (left_m/left), (right_m/right)
	
def computeOtsuThreshold(edgeNbdList, ccList):
	ccSet = set(ccList)
	maxVarBetween = 0
	finalT = (max(ccSet) + min(ccSet))/2
	for T in ccSet:
		nOT, nBT, miO_T, miB_T = getPixelCountsAndAvg(T, ccList)
		varBetween = nOT*nBT*(miO_T-miB_T)*(miO_T-miB_T)
		if maxVarBetween < varBetween:
			maxVarBetween = varBetween
			finalT = T

	return finalT

def computeMean(ParameterList):
	L = []
	for pl in ParameterList:
		if len(pl) != 0:
			L.append(sum(pl)/len(pl))
		else:
			L.append(0)

	return L[0], L[1], L[2], L[3], L[4], L[5], L[6]

def computeMeanNbdProp(G,edge):
	v1 = edge[0]
	v2 = edge[1]
	nbd_v1 = set(G.neighbors(v1))
	nbd_v2 = set(G.neighbors(v2))
	nodeSet = nbd_v1.union(nbd_v2)
	#nodeSet = nodeSet.difference(set((v1,v2)))
	edgeCount = 0
	totJI = totCC = totMCC = totUCC = totLCC = totRCC = 0
	for x in nodeSet:
		for y in nodeSet:			
			if G.has_edge(x,y):
				Lcc, Rcc, UMC = ComputeOnlyNbd(G, (x,y))
				totJI += computeJI(G, (x,y))
				totCC += computeCC(G, (x,y))
				totMCC += computeModifiedCC(G, (x,y))
				totUCC += computeUncommonCC(G, (x,y))
				totLCC += Lcc
				totRCC += Rcc
				edgeCount += 1
	return totJI/edgeCount, totCC/edgeCount, totMCC/edgeCount, totUCC/edgeCount, totLCC/edgeCount, totRCC/edgeCount


########################### computing threshold - start ###########################

########################### creating the meta graph - start ###########################
def getMetaGraph(G, edgeList):
	#input is a list of sets
	superNodes = []
	tempG = nx.Graph()

	for edge in edgeList:
		tempG.add_edge(edge[0], edge[1])

	#nx.draw_networkx(tempG)
	#plt.show()
	
	'''
	#connect the edges arrording to G
	for i in list(tempG.nodes):
		for j in list(tempG.nodes):
			if G.has_edge(i, j):
				tempG.add_edge(i,j)

	'''
	#collect the components as super nodes
	for c in nx.connected_components(tempG):
		superNodes.append(set(c))


	mergedSupNodes = set([])
	for supNode in superNodes:
		mergedSupNodes = mergedSupNodes.union(supNode)

	allNodes =  set(G.nodes())

	#create rest graphs
	G1 = nx.Graph()
	for edge in G.edges():
		if edge[0] not in mergedSupNodes and edge[1] not in mergedSupNodes:
			G1.add_edge(edge[0], edge[1])


	#create super nodes and connect the super nodes with the rest nodes
	for supNode in superNodes:
		min_label = min(supNode)
		G1.add_node(min_label)
		restNodes = allNodes.difference(mergedSupNodes)
		for edge in G.edges():
			if edge[0] in supNode and edge[1] in restNodes:
				G1.add_edge(min_label, edge[1])
			elif edge[1] in supNode and edge[0] in restNodes:
				G1.add_edge(min_label, edge[0])


	#connect the super nodes
	for supNode1 in superNodes:
		for supNode2 in superNodes:
			if(supNode1 != supNode2):
				for i in supNode1:
					for j in supNode2:
						if G.has_edge(i, j):
							G1.add_edge(min(supNode1), min(supNode2))

	return G1
########################### creating the meta graph - end ###########################
########################### Trimming a graph - start ###########################
def TrimGraph(G, core):
	H = G.__class__()
	H.add_nodes_from(G)
	H.add_edges_from(G.edges)
	while True:
		coreNodeList = []
		for node in H.nodes():
			if H.degree[node] == core:
				coreNodeList.append(node)
		if len(coreNodeList) == 0:
			break
		else:
			for n in coreNodeList:
				H.remove_node(n)
	return H
########################### Trimming a graph - end ###########################
def getLabelForGCN(G, CC):
	comset = set([])
	nodeList = []
	labelList = []
	label = 0
	for com in CC:
		comset = comset.union(set(com))
		nodeList.append(com[0])
		nodeList.append(com[1])
		labelList.append(label)
		labelList.append(label)
		label += 1
	allnodes =  set(G.nodes())
	non_consNode = allnodes.difference(comset)
	print('community label range = 0 to ',label-1)
	nclabel = label
	for node in non_consNode:
		nodeList.append(node)
		labelList.append(label)
		label += 1
	print('non-community label range = ',nclabel, ' to ', label-1)
	print('Node List: ',nodeList)
	print('Label List: ',labelList)
	return nodeList, labelList

#deg vs clus-coeff
def getLabelForGCN2(G, CC):
	comset = set([])
	nodeList = []
	labelList = []
	label = 1
	#print('comm label = ',label)
	for com in CC:
		comset = comset.union(set(com))
		nodeList.append(com[0])
		nodeList.append(com[1])
		labelList.append(label)
		labelList.append(label)

	allnodes =  set(G.nodes())
	non_consNode = allnodes.difference(comset)
	label = 0
	#print('non-comm label = ',label)
	for node in non_consNode:
		nodeList.append(node)
		labelList.append(label)
	#nodeList[0], nodeList[len(nodeList)-1] = nodeList[len(nodeList)-1], nodeList[0] 
	#labelList[0], labelList[len(nodeList)-1] = labelList[len(nodeList)-1], labelList[0] 
	print('Node List: ',nodeList)
	print('Label List: ',labelList)
	return nodeList, labelList
	
#clus-coeff
def getLabelForGCN3(G):
	node_clus_sequence = nx.clustering(G)
	no_of_nodes = G.number_of_nodes()
	nodeList = []
	labelList = []
	N100 = []
	N0 = []
	NM = []
	L100 = []
	L0 = []
	LM = []
	for n in node_clus_sequence:
		if node_clus_sequence[n] <= 1.0 and node_clus_sequence[n] >= 0.6:
			N100.append(n)
			L100.append(2)
		elif  node_clus_sequence[n] <= 0.3 and node_clus_sequence[n] >= 0.0:
			N0.append(n)
			L0.append(0)	
		elif  node_clus_sequence[n] <= 0.6 and node_clus_sequence[n] >= 0.4:
			NM.append(n)
			LM.append(1)

	for i in range(int(len(N0)*0.5)):
		nodeList.append(N0[i])
		labelList.append(L0[i])

	for i in range(0, int(len(NM)*0.5)):
		nodeList.append(NM[i])
		labelList.append(LM[i])

	for i in range(0,int(len(N100)*0.5)):
		nodeList.append(N100[i])
		labelList.append(L100[i])

	print('Node List: ',nodeList)
	print('Label List: ',labelList)
	return nodeList, labelList

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
			b100.append(edgeDic[edge])
		else:
			b0.append(edgeDic[edge])
	return b0, b100

def getIndex(u, comlist):
	for com in comlist:
		if u in com:
			return comlist.index(com)

def computeB0B100Fortunato(G, comlist, edge_dic):
	b0 = []
	b100 = []
	B0 = []
	B100 = []
	GT0 = []
	GT100 = []
	#print(G.edges())
	for e in G.edges():
		if e[0] < e[1]:
			edge = e
		else:
			edge = (e[1], e[0])

		u = G.nodes[edge[0]]['name'] 
		v = G.nodes[edge[1]]['name']
		#print(e,(u,v))
		#print(e, getIndex(u, comlist), getIndex(v, comlist))
		if getIndex(u, comlist) == getIndex(v, comlist):
			b100.append(e)
			B100.append((u,v))
		else:
			#print(edge)
			b0.append(e)
			B0.append((u,v))

		if edge_dic[e] < no_of_perm:
			GT0.append((u,v))
		else:
			GT100.append((u,v))
	return b0, b100, B0, B100, GT0, GT100


def computePRF_GCN(b0, b100, B0, B100, edge_dic):
	CM = [[0,0],[0,0]]
	P_0_0, P_0_100 = computeCM2(b0, B0, 0, CM, edge_dic)
	P_100_0, P_100_100 = computeCM2(b100, B100, 1, CM, edge_dic)
	
	return ComputePRA2(CM)

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

####################code to compute purturbation - start###########################
def saveComm(pred_comm, j, i):
	fname = "partitionedGraph/"+str(j)+str(i)+".txt"
	fileW = open(fname,"wb")
	pickle.dump(pred_comm, fileW)
	fileW.close()
	return fname
		
def loadPickleFile(f):
	fileR = open(f,"rb")
	mycomm = pickle.load(fileR)
	return fileR, mycomm

def compareComm(fileList):
	import statistics as stat
	NMI_table = []
	for f1 in fileList:
		fileR1, comm1 = loadPickleFile(f1)
		for f2 in fileList:
			if f1 != f2:
				fileR2, comm2 = loadPickleFile(f2)
				NMI_table.append(findNMI(comm1, comm2))
				fileR2.close()
		fileR1.close()
	#print('Mean NMI: ',sum(NMI_table)/len(NMI_table))
	#print('Variance NMI: ',sum(NMI_table)/len(NMI_table))
	print('Mean NMI: ',stat.mean(NMI_table))
	print('Variance NMI: ',stat.variance(NMI_table))

def getCDISNodeList(G, comm):
	conn = []
	dis_conn = []
	for c1 in comm:
		for c2 in comm:
			if G.has_edge(c1,c2):
				conn.append((c1,c2))
			else:
				dis_conn.append((c1,c2))
	return conn, dis_conn

def makePurturbation2(G, commwise_connList, commwise_dis_connList):
	for comm_conn, comm_disconn in zip(commwise_connList, commwise_dis_connList): 
		for i in range( int(min(len(comm_conn), len(comm_disconn))) ):
			if G.has_edge(*comm_conn[i]):
				G.remove_edge(*comm_conn[i])
			G.add_edge(*comm_disconn[i])
	return G

def getPerturbGraph2(G, comList):
	commwise_connList = []
	commwise_dis_connList = []
	for comm in comList:
		conn, dis_conn = getCDISNodeList(G, comm)
		commwise_connList.append(conn)
		commwise_dis_connList.append(dis_conn)
	G = makePurturbation(G, commwise_connList, commwise_dis_connList)
		
	return G

def makePurturbation(G, connList, dis_connList):
	import random
	random.shuffle(connList)
	random.shuffle(dis_connList)
	for i in range( int(min(len(connList), len(dis_connList))*0.1) ):
		if G.has_edge(*connList[i]):
			G.remove_edge(*connList[i])
		G.add_edge(*dis_connList[i])
	return G

def getPerturbGraph(G, comList):
	conn = []
	dis_conn = []
	for comm in comList:
		conn_temp, dis_conn_temp = getCDISNodeList(G, comm)
		conn.extend(conn_temp)
		dis_conn.extend(dis_conn_temp)
	G = makePurturbation(G, conn, dis_conn)
	fname = "temp.txt"
	nx.write_edgelist(G, fname)	
	return fname
	

####################code to compute purturbation - start###########################

def getAvg(mylist):
	return sum(mylist)/len(mylist)
