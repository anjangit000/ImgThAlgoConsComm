from __future__ import division
from common import *
from intbitset import *
import networkx as nx
import community as louvain
import numpy as np
import otsu
import array as arr
import copy
########################### Precision,Recall and Accuracy - start ###########################

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

def computeCommon(nbd_v1, nbd_v2):
    my_dic1 = {n1:0 for n1 in nbd_v1}
    common = []
    for n2 in nbd_v2:
        if n2 in my_dic1:
            common.append(n2)
    return common


def JI(set_union_len, set_intersection_len):#This is jaccard index
	if set_union_len == 0:
		return 0
	return set_intersection_len/set_union_len

#method to compute a JI for a given edge
def computeJI(set_union_len, set_intersection_len):
	return JI(set_union_len, set_intersection_len)

def computeCC(edgeCount, outdegree):
	num = edgeCount
	#print(num)
	deno = outdegree*(outdegree-1)
	if outdegree == 0 or outdegree == 1:
		return 0
	return 2*num/deno

def computeModifiedCC(edgeCount, outdegree):
	num = edgeCount
	deno = outdegree*(outdegree-1)
	if outdegree == 0 or outdegree == 1:
		return 0
	return 2*num/deno


def computeUncommonCC2(total, inter, left, right, leftNode, rightNode, middle):
	edgeCount = total - left - right + inter
	outdegree = (leftNode-middle)*(rightNode-middle)
	num = edgeCount
	deno = outdegree
	if outdegree == 0 or outdegree == 1:
		return 0
	return num/deno


def Makecounter(myDic, myKey):
		if myKey in myDic:
			myDic[myKey] += 1
		else:
			myDic[myKey] = 1


#nbd_dic = {}

def getNbd(G, n, capacity, nbd_dic):
	if n not in nbd_dic:
		nbd = intbitset([n2 for n2 in G.neighbors(n)])
		if len(nbd_dic) < capacity:
			nbd_dic[n] = nbd
	else:
		nbd = nbd_dic[n]
	return nbd

def getEdge(e):
	if e[0] < e[1]:
		edge = e
	else:
		edge = (e[1], e[0])
	return edge

def computeTriangle(v1, v2, node_triangle_dic, no_of_triangles):
	if v1 in node_triangle_dic:
		node_triangle_dic[v1] += no_of_triangles
	else:
		node_triangle_dic[v1] = no_of_triangles

	if v2 in node_triangle_dic:
		node_triangle_dic[v2] += no_of_triangles
	else:
		node_triangle_dic[v2] = no_of_triangles

'''
def computeParameters2(G, edge_dic):
	capacity = 20000
	buffer_cap = 5000
	node_deg = dict(G.degree())
	node_triangle_dic = {}
	JIdic = {}
	CCdic = {}
	Mccdic = {}
	Tridic = {}
	B0 = []
	B100 = []
	Orig0 = []
	Orig100 = []
	nbd_dic = {}
	Rest_edges1 = []
	v = -1
	s = {}
	edge_prop_dic = {}
	i = 0
	prev_inter = intbitset([])
	prev_k1 = 0

	count = 0
	for e in G.edges():
		count += 1
		#if count%10 == 0 or (count+1)%10 == 0 or (count+2)%10 == 0:
		#	continue
		e = getEdge(e)
		v1 = e[0]
		v2 = e[1]



		if node_deg[v1] == 1 or node_deg[v2] == 1:
			B100.append(e)
			computeTriangle(v1, v2, node_triangle_dic, 0)
			edge_prop_dic[e] = arr.array('d',[0, 0, 0, 0, node_deg[v1]+node_deg[v2]-2])
			Makecounter(JIdic, 0)
			Makecounter(CCdic, 0)
			continue


		if len(nbd_dic) >= capacity and v1 not in nbd_dic:
			if v != v1:
				nbd_v1 = intbitset([n1 for n1 in G.neighbors(v1)])
				s = nbd_v1
				v = v1
			else:
				nbd_v1 = s
		else:
			nbd_v1 = getNbd(G, v1, capacity, nbd_dic)
		nbd_v2 = getNbd(G, v2, capacity, nbd_dic)
		
		set_intersection = nbd_v1 & nbd_v2

		no_of_triangles = len(set_intersection)
		computeTriangle(v1, v2, node_triangle_dic, no_of_triangles)

		k1 = 0 
		for i1 in set_intersection:
			#if node_deg[i1] == 2:
				#continue
			nbd_i1 = getNbd(G, i1, capacity, nbd_dic)
			k1 += len(nbd_i1 & set_intersection)

		tot_nbd = node_deg[v1] + node_deg[v2] - no_of_triangles
		
		ji = round(computeJI(tot_nbd, no_of_triangles),2)
		#if ji >= 1:
			#print([n1 for n1 in G.neighbors(v1)], [n2 for n2 in G.neighbors(v2)], no_of_triangles)
		cc = round(computeCC(k1//2, no_of_triangles),2)

		Makecounter(JIdic, ji)
		Makecounter(CCdic, cc)

		#print(e, tot_nbd)

		edge_prop_dic[e] = arr.array('d',[ji, cc, k1, no_of_triangles, tot_nbd-no_of_triangles-2])
		
		
		#if ji == 0 and node_deg[v2] > 2 and node_deg[v1] > 2:
			#B0.append(e)
		
		if ji > 0 and (node_deg[v2] == 2 or node_deg[v1] == 2):
			B100.append(e)	
		else:
			Rest_edges1.append(e)
		
		#print(e,k1, set_intersection)
		i += 1
		#Rest_edges1.append(e)

	count = 0
	for e in G.edges():
		count += 1
		#if count%10 == 0 or (count+1)%10 == 0 or (count+2)%10 == 0:
		#	continue
		e = getEdge(e)
		left = node_triangle_dic[e[0]]
		right = node_triangle_dic[e[1]]
		inter = edge_prop_dic[e][2]
		num = (left + right - inter)//2 + edge_prop_dic[e][4]
		den = (node_deg[e[0]] + node_deg[e[1]] - edge_prop_dic[e][3])
		mcc = round(computeModifiedCC(num, den),2)
		
		Makecounter(Mccdic, mcc)
		edge_prop_dic[e].append(mcc)
		den = (left+right-inter)
		if den == 0:
			tri_ratio = 0
		else:
			tri_ratio = round(inter/den,2)
		Makecounter(Tridic, tri_ratio)
		edge_prop_dic[e].append(tri_ratio)
		
	return JIdic, CCdic, Mccdic, Tridic, B0, B100, Rest_edges1, edge_prop_dic
'''
		
def computeParameters2(G, edge_dic):
	capacity = 20000
	buffer_cap = 5000
	node_deg = dict(G.degree())
	node_triangle_dic = {}
	JIdic = {}
	CCdic = {}
	Mccdic = {}
	Tridic = {}
	B0 = []
	B100 = []
	Orig0 = []
	Orig100 = []
	nbd_dic = {}
	Rest_edges1 = []
	v = -1
	s = {}
	edge_prop_dic = {}
	i = 0
	for e in G.edges():
		v1 = e[0]
		v2 = e[1]
		e = getEdge(e)
		if edge_dic[e] == no_of_perm:
			Orig100.append(e)
		else:
			Orig0.append(e)


		if node_deg[v1] == 1 or node_deg[v2] == 1:
			B100.append(e)
			computeTriangle(v1, v2, node_triangle_dic, 0)
			edge_prop_dic[e] = arr.array('d',[0, 0, 0, 0, node_deg[v1]+node_deg[v2]-1])
			Makecounter(JIdic, 0)
			Makecounter(CCdic, 0)
			continue


		if len(nbd_dic) >= capacity and v1 not in nbd_dic:
			if v != v1:
				nbd_v1 = intbitset([n1 for n1 in G.neighbors(v1)])
				s = nbd_v1
				v = v1
			else:
				nbd_v1 = s
		else:
			nbd_v1 = getNbd(G, v1, capacity, nbd_dic)
		nbd_v2 = getNbd(G, v2, capacity, nbd_dic)
		
		set_intersection = nbd_v1 & nbd_v2

		no_of_triangles = len(set_intersection)
		computeTriangle(v1, v2, node_triangle_dic, no_of_triangles)

		k1 = 0 
		for i1 in set_intersection:
			#if node_deg[i1] == 2:
				#continue
			nbd_i1 = getNbd(G, i1, capacity, nbd_dic)
			k1 += len(nbd_i1 & set_intersection)


 		
		ji = round(computeJI((node_deg[v1] + node_deg[v2] - no_of_triangles - 1), no_of_triangles),2)
		#if ji >= 1:
			#print([n1 for n1 in G.neighbors(v1)], [n2 for n2 in G.neighbors(v2)], no_of_triangles)
		cc = round(computeCC(k1//2, no_of_triangles),2)
		common_tri = no_of_triangles

		Makecounter(JIdic, ji)
		Makecounter(CCdic, cc)


		edge_prop_dic[e] = arr.array('d',[ji, cc, k1, common_tri, node_deg[v1]+node_deg[v2]-common_tri-1])
		
		
		#if ji == 0 and node_deg[v2] > 2 and node_deg[v1] > 2:
			#B0.append(e)
		
		if ji > 0 and (node_deg[v2] == 2 or node_deg[v1] == 2):
			B100.append(e)	
		else:
			Rest_edges1.append(e)
		
		#print(e,i)
		i += 1
		#Rest_edges1.append(e)

	for e in G.edges():
		e = getEdge(e)
		left = node_triangle_dic[e[0]]
		right = node_triangle_dic[e[1]]
		inter = edge_prop_dic[e][2]
		num = (left + right - inter)//2 + edge_prop_dic[e][4]
		den = (node_deg[e[0]] + node_deg[e[1]] - edge_prop_dic[e][3])
		mcc = round(computeModifiedCC(num, den),2)
		Makecounter(Mccdic, mcc)
		edge_prop_dic[e].append(mcc)
		den = (left+right-inter)
		if den == 0:
			tri_ratio = 0
		else:
			tri_ratio = round(inter/den,2)
		Makecounter(Tridic, tri_ratio)
		edge_prop_dic[e].append(tri_ratio)

	return JIdic, CCdic, Mccdic, Tridic, B0, B100, Rest_edges1, edge_prop_dic


########################### computing the parameters - end ###########################

########################### computing the Filter class - start ###########################
def plotBar(propDic, xlabel='D_both', ylabel='frequency', color='r', threshold=0.5):
    import matplotlib.pyplot as plt
    x = [] 
    y = []
    for key in (sorted(propDic)):
        x.append(key*10)
        y.append(propDic[key])

    fig, ax = plt.subplots()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.bar(x, y, width=0.2, color=color)
    ax.axvline(x=threshold*10, color='k')
    plt.show()

def plotBar2(propDic1, propDic2, propDic3, propDic4, th1, th2, th3, th4):
    import matplotlib.pyplot as plt
    x1 = []; x2 = []; x3 = []; x4 = []; y1 = []; y2 = []; y3 = []; y4 = []
    for key in (sorted(propDic1)):
        x1.append(key*10)
        y1.append(propDic1[key])
    for key in (sorted(propDic2)):
        x2.append(key*10)
        y2.append(propDic2[key])
    for key in (sorted(propDic3)):
        x3.append(key*10)
        y3.append(propDic3[key])
    for key in (sorted(propDic4)):
        x4.append(key*10)
        y4.append(propDic4[key])

    ylabel = 'frequency'
    ax = plt.subplot(2,2,1)
    ax.set_xlabel('D_both')
    ax.set_ylabel(ylabel)
    ax.bar(x1, y1, width=0.2, color='r')
    ax.axvline(x=th1*10, color='k')

    ax = plt.subplot(2,2,2)
    ax.set_xlabel('D_any')
    ax.set_ylabel(ylabel)
    ax.bar(x2, y2, width=0.2, color='g')
    ax.axvline(x=th2*10, color='k')

    ax = plt.subplot(2,2,3)
    ax.set_xlabel('D_tri')
    ax.set_ylabel(ylabel)
    ax.bar(x3, y3, width=0.2, color='b')
    ax.axvline(x=th3*10, color='k')

    ax = plt.subplot(2,2,4)
    ax.set_xlabel('JI')
    ax.set_ylabel(ylabel)
    ax.bar(x4, y4, width=0.2, color='c')
    ax.axvline(x=th4*10, color='k')

    plt.show()


def computeFilteredBinaryOtsu(G, edge_dic):
	JIdic, CCdic, Mccdic, Tridic, B0, B100, Rest_edges, edge_prop_dic = computeParameters2(G, edge_dic)
	T_ji_otsu = computeOtsuThreshold2(JIdic)
	T_cc_otsu = computeOtsuThreshold2(CCdic)
	T_mcc_otsu = computeOtsuThreshold2(Mccdic)
	T_tri_otsu = computeOtsuThreshold2(Tridic)
	print(T_ji_otsu, T_cc_otsu, T_mcc_otsu, T_tri_otsu)
	#plotBar(CCdic, xlabel = 'D_both', ylabel = 'frequency', color = 'r', threshold = T_cc_otsu)
	#plotBar2(CCdic, Mccdic, Tridic, JIdic, T_cc_otsu, T_mcc_otsu, T_tri_otsu, T_ji_otsu)
	CM = [[0,0],[0,0]]
	
	for e in Rest_edges:
		e = getEdge(e)
		ji = edge_prop_dic[e][0]
		cc = edge_prop_dic[e][1]
		mcc = edge_prop_dic[e][5]
		tri_ratio = edge_prop_dic[e][6]
		if (ji >= T_ji_otsu/2 and cc > T_cc_otsu/2 ) or mcc >= T_mcc_otsu or tri_ratio > T_tri_otsu/2:
			B100.append(e)
		else:
			B0.append(e)

	computeCM2(B0, 0, CM, edge_dic)
	computeCM2(B100, 1, CM, edge_dic)

	ComputePRA2(CM)

	return B0, B100

def getGraphFromB(G, B):
	G1 = G.__class__()
	G1.add_nodes_from(G)
	G1.add_edges_from(G.edges)
	for e in B:
		if G1.has_node(e[0]):
			G1.remove_node(e[0])
		if G1.has_node(e[1]):
			G1.remove_node(e[1])
	return G1

	



def computeFilteredHist_multiOtsu(G, edge_dic, fileName):
	JIdic, CCdic, Mccdic, Tridic, node_triangle_dic, B0, B100, Orig0, Orig100, Rest_edges, edge_prop_dic = computeParameters2(G, edge_dic)
	#print(JIdic, CCdic, Mccdic, Tridic)
	T_ji_otsu = computeMultiOtsuThreshold(JIdic)[0]
	T_cc_otsu = computeMultiOtsuThreshold(CCdic)[0]
	T_mcc_otsu = computeMultiOtsuThreshold(Mccdic)[0]
	T_tri_otsu = computeMultiOtsuThreshold(Tridic)[0]
	print(T_ji_otsu, T_cc_otsu, T_mcc_otsu, T_tri_otsu)
	CM = [[0,0],[0,0]]
	
	for e in Rest_edges:
		ji = edge_prop_dic[e][0]
		cc = edge_prop_dic[e][1]
		mcc = edge_prop_dic[e][5]
		tri_ratio = edge_prop_dic[e][6]
		if (ji >= T_ji_otsu/2 and cc > T_cc_otsu/2) or mcc >= T_mcc_otsu or tri_ratio > T_tri_otsu/2:
			B100.append(e)
		else:
			B0.append(e)

	'''
	print('Ground truth\n', edge_dic)
	print('\n')
	print('B100\n',B100)
	'''
		
	#print('(',edge[0], edge[1],')', ',', ji, ',', cc, ',', mcc, ',', ucc, ',', edge_dic[edge])
	print('********************PREDICTED********************')
	
	print('Belongingness 0: ',len(B0))
	print('Belongingness 100: ',len(B100))


	print('********************ACTUAL********************')
	print('Belongingness 0: ',len(Orig0))
	print('Belongingness 100: ',len(Orig100))

	print('*********************************************')
	computeCM2(B0, 0, CM, edge_dic)
	computeCM2(B100, 1, CM, edge_dic)

	ComputePRA2(CM)

	#print(edge_dic)
	#print(B100)
	return B0, B100, Orig0, Orig100




def getCumTupDic(Prop_freq_dic, flag=True):
	s = 0
	tup = []
	for k in sorted(Prop_freq_dic, reverse=flag):
		s += Prop_freq_dic[k]
		tup.append((k, s))	
	return dict(tup)


def getHalfWayMapped(Prop_freq_dic, th_list, val):
	a_list = [k for k in sorted(Prop_freq_dic, reverse=True)]
	th_list_mapped = {}
	for t in th_list:
		x = round(t/val,2)
		absolute_difference_function = lambda list_value : abs(list_value - x)
		closest_value = min(a_list, key=absolute_difference_function)
		th_list_mapped[t] = closest_value
	return th_list_mapped



def getB100FromComm(G):
	B100_comm = []
	partition = louvain.best_partition(G)
	for edge in G.edges():
		if partition[edge[0]] == partition[edge[1]]:
			B100_comm.append(edge)
	return B100_comm

def computeFilteredHist_multiOtsu_best2(G, edge_dic):
	B100_comm = getB100FromComm(G)
	b100_com_len = len(B100_comm)
	b0_comm_len = G.size() - b100_com_len
	JIdic, CCdic, Mccdic, Tridic, B0, B100, Rest_edges, edge_prop_dic = computeParameters2(G, edge_dic)
	JIdic_cum = getCumTupDic(JIdic); CCdic_cum = getCumTupDic(CCdic); Mccdic_cum = getCumTupDic(Mccdic);Tridic_cum = getCumTupDic(Tridic)
	T_ji_otsu_list =  computeMultiOtsuThreshold(JIdic)
	T_cc_otsu_list =   computeMultiOtsuThreshold(CCdic)
	T_mcc_otsu_list =   computeMultiOtsuThreshold(Mccdic)
	T_tri_otsu_list =  computeMultiOtsuThreshold(Tridic)

	T_ji_otsu = computeOtsuThreshold2(JIdic)
	T_cc_otsu = computeOtsuThreshold2(CCdic)
	T_mcc_otsu = computeOtsuThreshold2(Mccdic)
	T_tri_otsu = computeOtsuThreshold2(Tridic)

	dic1 = getHalfWayMapped(JIdic, T_ji_otsu_list, 2)
	dic2 = getHalfWayMapped(CCdic, T_cc_otsu_list, 2)
	dic3 = getHalfWayMapped(Mccdic, T_mcc_otsu_list, 1)
	dic4 = getHalfWayMapped(Tridic, T_tri_otsu_list, 2)
	j = 0
	c = []
	prop_list = [(T_ji_otsu, T_cc_otsu, T_mcc_otsu, T_tri_otsu)]
	g_size = G.size()

	B_100_len = len(B100)
	for T_ji_otsu in T_ji_otsu_list:
		for T_cc_otsu in T_cc_otsu_list:
			for T_mcc_otsu in T_mcc_otsu_list:
				for T_tri_otsu in T_tri_otsu_list:
					prop_list.append((T_ji_otsu, T_cc_otsu, T_mcc_otsu, T_tri_otsu))
					m1 = dic1[T_ji_otsu]
					m2 = dic2[T_cc_otsu]
					m3 = dic3[T_mcc_otsu]
					m4 = dic4[T_tri_otsu]
					
					b100_len = max(min(JIdic_cum[m1],CCdic_cum[m2]),Mccdic_cum[m3],Tridic_cum[m4])
					b0_len = g_size-b100_len
					j += 1
					
					if b0_len == 0:
						c.append(b100_len)
					else:
						c.append(abs(1-(b100_len/b0_len)))

	min_c = [c.index(min(c))]

	F1_100_list = []
	for i in min_c:
		CM = [[0,0],[0,0]]
		B_100 = []
		B_0 = []
		T_ji_otsu = prop_list[i][0]; T_cc_otsu = prop_list[i][1]; T_mcc_otsu = prop_list[i][2]; T_tri_otsu = prop_list[i][3]
		

		for e in Rest_edges:
			e = getEdge(e)
			ji = edge_prop_dic[e][0]
			cc = edge_prop_dic[e][1]
			mcc = edge_prop_dic[e][5]
			tri_ratio = edge_prop_dic[e][6]
			if (ji >= T_ji_otsu/2 and cc > T_cc_otsu/2) or mcc >= T_mcc_otsu or tri_ratio > T_tri_otsu/2:
				B_100.append(e)
			else:
				B_0.append(e)

		B_100 = B_100 + B100
		
		#print('Ground truth\n', edge_dic)
		#print('\n')
		#print('B100\n',B_100)
		#print('(',edge[0], edge[1],')', ',', ji, ',', cc, ',', mcc, ',', ucc, ',', edge_dic[edge])
		print('********************PREDICTED********************')
		
		print('Belongingness 0: ',len(B_0))
		print('Belongingness 100: ',len(B_100))


		#print('********************ACTUAL********************')
		#print('Belongingness 0: ',len(Orig0))
		#print('Belongingness 100: ',len(Orig100))

		print('*********************************************')
		computeCM2(B_0, 0, CM, edge_dic)
		computeCM2(B_100, 1, CM, edge_dic)

		f1_100 = ComputePRA2(CM)
		F1_100_list.append(f1_100)
		print('--------------------------------------\n')

	return B_0, B_100
	


def multiOtsuCommonBest2(G, edge_dic):
	JIdic, CCdic, Mccdic, Tridic, B0, B100, Rest_edges, edge_prop_dic = computeParameters2(G, edge_dic)
	JIdic_cum = getCumTupDic(JIdic); CCdic_cum = getCumTupDic(CCdic); Mccdic_cum = getCumTupDic(Mccdic);Tridic_cum = getCumTupDic(Tridic)
	#print(dict([(k, CCdic[k]) for k in sorted(CCdic, reverse=True)]))
	#print('\n')
	#print(CCdic_cum)

	T_ji_otsu_list =  computeMultiOtsuThreshold(JIdic)
	T_cc_otsu_list =   computeMultiOtsuThreshold(CCdic)
	T_mcc_otsu_list =   computeMultiOtsuThreshold(Mccdic)
	T_tri_otsu_list =  computeMultiOtsuThreshold(Tridic)

	T_ji_otsu = computeOtsuThreshold2(JIdic)
	T_cc_otsu = computeOtsuThreshold2(CCdic)
	T_mcc_otsu = computeOtsuThreshold2(Mccdic)
	T_tri_otsu = computeOtsuThreshold2(Tridic)

	dic1 = getHalfWayMapped(JIdic, T_ji_otsu_list, 2)
	dic2 = getHalfWayMapped(CCdic, T_cc_otsu_list, 2)
	dic3 = getHalfWayMapped(Mccdic, T_mcc_otsu_list, 1)
	dic4 = getHalfWayMapped(Tridic, T_tri_otsu_list, 2)
	#print(dic3)
	#print(T_ji_otsu_list)
	#print(T_cc_otsu_list)
	j = 0
	c = []
	prop_list = [(T_ji_otsu, T_cc_otsu, T_mcc_otsu, T_tri_otsu)]
	g_size = G.size()

	B_100_len = len(B100)
	for T_ji_otsu in T_ji_otsu_list:
		for T_cc_otsu in T_cc_otsu_list:
			for T_mcc_otsu in T_mcc_otsu_list:
				for T_tri_otsu in T_tri_otsu_list:
					prop_list.append((T_ji_otsu, T_cc_otsu, T_mcc_otsu, T_tri_otsu))
					m1 = dic1[T_ji_otsu]; m2 = dic2[T_cc_otsu]; m3 = dic3[T_mcc_otsu];	m4 = dic4[T_tri_otsu]
					
					b100_len = max(min(JIdic_cum[m1],CCdic_cum[m2]),Mccdic_cum[m3],Tridic_cum[m4])
					b0_len = g_size-b100_len
					#print(j, b100_len, b0_len)
					j += 1
					
					if b0_len == 0:
						c.append(b100_len)
					else:
						c.append(abs(1-(b100_len/b0_len)))

	i = c.index(min(c))

	CM = [[0,0],[0,0]]
	B_100 = []
	B_0 = []
	T_ji_otsu = prop_list[i][0]; T_cc_otsu = prop_list[i][1]; T_mcc_otsu = prop_list[i][2]; T_tri_otsu = prop_list[i][3]
	
	#print(T_ji_otsu, T_cc_otsu, T_mcc_otsu, T_tri_otsu)
	for e in Rest_edges:
		e = getEdge(e)
		ji = edge_prop_dic[e][0]
		cc = edge_prop_dic[e][1]
		mcc = edge_prop_dic[e][5]
		tri_ratio = edge_prop_dic[e][6]
		if (ji >= T_ji_otsu/2 and cc > T_cc_otsu/2) or mcc >= T_mcc_otsu or tri_ratio > T_tri_otsu/2:# or (mcc >= 0.43 and cc == 1):
			B100.append(e)
		else:
			B0.append(e)

	#print(T_ji_otsu, T_cc_otsu, T_mcc_otsu, T_tri_otsu)
	return B0, B100, edge_prop_dic

def multiOtsuCommonBest2_keepOld(G, edge_dic, edge_prop_dic_old):
	JIdic, CCdic, Mccdic, Tridic, node_triangle_dic, B0, B100, Orig0, Orig100, Rest_edges, edge_prop_dic = computeParameters2(G, edge_dic)
	JIdic_cum = getCumTupDic(JIdic); CCdic_cum = getCumTupDic(CCdic); Mccdic_cum = getCumTupDic(Mccdic);Tridic_cum = getCumTupDic(Tridic)

	T_ji_otsu_list =  computeMultiOtsuThreshold(JIdic)
	T_cc_otsu_list =   computeMultiOtsuThreshold(CCdic)
	T_mcc_otsu_list =   computeMultiOtsuThreshold(Mccdic)
	T_tri_otsu_list =  computeMultiOtsuThreshold(Tridic)

	T_ji_otsu = computeOtsuThreshold2(JIdic)
	T_cc_otsu = computeOtsuThreshold2(CCdic)
	T_mcc_otsu = computeOtsuThreshold2(Mccdic)
	T_tri_otsu = computeOtsuThreshold2(Tridic)

	dic1 = getHalfWayMapped(JIdic, T_ji_otsu_list, 2)
	dic2 = getHalfWayMapped(CCdic, T_cc_otsu_list, 2)
	dic3 = getHalfWayMapped(Mccdic, T_mcc_otsu_list, 1)
	dic4 = getHalfWayMapped(Tridic, T_tri_otsu_list, 2)
	#print(dic3)
	#print(T_ji_otsu_list)
	#print(T_cc_otsu_list)
	j = 0
	c = []
	prop_list = [(T_ji_otsu, T_cc_otsu, T_mcc_otsu, T_tri_otsu)]
	g_size = G.size()

	B_100_len = len(B100)
	for T_ji_otsu in T_ji_otsu_list:
		for T_cc_otsu in T_cc_otsu_list:
			for T_mcc_otsu in T_mcc_otsu_list:
				for T_tri_otsu in T_tri_otsu_list:
					prop_list.append((T_ji_otsu, T_cc_otsu, T_mcc_otsu, T_tri_otsu))
					m1 = dic1[T_ji_otsu]; m2 = dic2[T_cc_otsu]; m3 = dic3[T_mcc_otsu];	m4 = dic4[T_tri_otsu]
					
					b100_len = max(min(JIdic_cum[m1],CCdic_cum[m2]),Mccdic_cum[m3],Tridic_cum[m4])
					b0_len = g_size-b100_len
					#print(j, b100_len, b0_len)
					j += 1
					
					if b0_len == 0:
						c.append(b100_len)
					else:
						c.append(abs(1-(b100_len/b0_len)))

	i = c.index(min(c))

	CM = [[0,0],[0,0]]
	B_100 = []
	B_0 = []
	T_ji_otsu = prop_list[i][0]; T_cc_otsu = prop_list[i][1]; T_mcc_otsu = prop_list[i][2]; T_tri_otsu = prop_list[i][3]

	for e in Rest_edges:
		e = getEdge(e)
		ji = edge_prop_dic_old[e][0]
		cc = edge_prop_dic_old[e][1]
		mcc = edge_prop_dic_old[e][5]
		tri_ratio = edge_prop_dic_old[e][6]
		if (ji >= T_ji_otsu/2 and cc > T_cc_otsu/2) or mcc >= T_mcc_otsu or tri_ratio > T_tri_otsu/2:
			B100.append(e)
		else:
			B0.append(e)

	return B0, B100

def getDistribution(B, edge_prop_dic):
	ji_dic = {}; cc_dic = {}; mcc_dic = {}; tri_dic = {}
	for e in B:
		e = getEdge(e)
		ji = edge_prop_dic[e][0]
		cc = edge_prop_dic[e][1]
		mcc = edge_prop_dic[e][5]
		tri_ratio = edge_prop_dic[e][6]
		Makecounter(ji_dic, ji)
		Makecounter(cc_dic, cc)
		Makecounter(mcc_dic, mcc)
		Makecounter(tri_dic, tri_ratio)
	print(([(k, ji_dic[k]) for k in sorted(ji_dic)]), '\n')
	print(([(k, cc_dic[k]) for k in sorted(cc_dic)]), '\n')
	print(([(k, mcc_dic[k]) for k in sorted(mcc_dic)]), '\n')
	print(([(k, tri_dic[k]) for k in sorted(tri_dic)]), '\n')

def computeFilteredHist_multiOtsu_repeatedBest2(G, edge_dic, fileName):
	B100_set = set([])
	G1 = nx.Graph()
	G1.add_edges_from(G.edges())
	B0, B100, edge_prop_dic_old = multiOtsuCommonBest2(G1, edge_dic)
	B100_set = B100_set.union(set(B100))

		
	for i in range(2):		
		G1 = nx.Graph()
		G1.add_edges_from(B0)
		B0, B100 = multiOtsuCommonBest2_keepOld(G1, edge_dic, edge_prop_dic_old)
		B100_set = B100_set.union(set(B100))

	B100 = list(B100_set)
	CM = [[0,0],[0,0]]
	computeCM2(B0, 0, CM, edge_dic)
	computeCM2(B100, 1, CM, edge_dic)

	ComputePRA2(CM)

	return B0, B100


	
def multiOtsuCommon(G, edge_dic):
	JIdic, CCdic, Mccdic, Tridic, B0, B100, Rest_edges = computeParameters2(G, edge_dic)
	
	#print([(k, Tridic[k]) for k in sorted(Tridic)])
	T_ji_otsu_list = computeMultiOtsuThreshold(JIdic)
	T_cc_otsu_list = computeMultiOtsuThreshold(CCdic)
	T_mcc_otsu_list = computeMultiOtsuThreshold(Mccdic)
	T_tri_otsu_list = computeMultiOtsuThreshold(Tridic)

	T_ji_otsu = max(T_ji_otsu_list)
	T_cc_otsu = max(T_cc_otsu_list)
	T_mcc_otsu = max(T_mcc_otsu_list)
	T_tri_otsu = max(T_tri_otsu_list)

	print(T_ji_otsu_list)
	print(T_cc_otsu_list)
	print(T_mcc_otsu_list)
	print(T_tri_otsu_list)
	for e in Rest_edges:
		ji = edge_prop_dic[e][0]
		cc = edge_prop_dic[e][1]
		mcc = edge_prop_dic[e][5]
		tri_ratio = edge_prop_dic[e][6]
		if (ji >= T_ji_otsu and cc > T_cc_otsu) or mcc >= T_mcc_otsu or tri_ratio > T_tri_otsu:
		#if (cc >= T_cc_otsu/2):
			B100.append(e)
		else:
			B0.append(e)

	edge_prop_dic_old = copy.deepcopy(edge_prop_dic)
	return B0, B100, edge_prop_dic_old

def getOptimumThres(G, JIdic, CCdic, Mccdic, Tridic, B0, B100):
	JIdic_cum = getCumTupDic(JIdic); CCdic_cum = getCumTupDic(CCdic); Mccdic_cum = getCumTupDic(Mccdic);Tridic_cum = getCumTupDic(Tridic)

	T_ji_otsu_list =  computeMultiOtsuThreshold(JIdic)
	T_cc_otsu_list =   computeMultiOtsuThreshold(CCdic)
	T_mcc_otsu_list =   computeMultiOtsuThreshold(Mccdic)
	T_tri_otsu_list =  computeMultiOtsuThreshold(Tridic)

	T_ji_otsu = computeOtsuThreshold2(JIdic)
	T_cc_otsu = computeOtsuThreshold2(CCdic)
	T_mcc_otsu = computeOtsuThreshold2(Mccdic)
	T_tri_otsu = computeOtsuThreshold2(Tridic)

	dic1 = getHalfWayMapped(JIdic, T_ji_otsu_list, 2)
	dic2 = getHalfWayMapped(CCdic, T_cc_otsu_list, 2)
	dic3 = getHalfWayMapped(Mccdic, T_mcc_otsu_list, 1)
	dic4 = getHalfWayMapped(Tridic, T_tri_otsu_list, 2)
	#print(dic3)
	#print(T_ji_otsu_list)
	#print(T_cc_otsu_list)
	j = 0
	c = []
	prop_list = [(T_ji_otsu, T_cc_otsu, T_mcc_otsu, T_tri_otsu)]
	g_size = G.size()

	B_100_len = len(B100)
	for T_ji_otsu in T_ji_otsu_list:
		for T_cc_otsu in T_cc_otsu_list:
			for T_mcc_otsu in T_mcc_otsu_list:
				for T_tri_otsu in T_tri_otsu_list:
					prop_list.append((T_ji_otsu, T_cc_otsu, T_mcc_otsu, T_tri_otsu))
					m1 = dic1[T_ji_otsu]; m2 = dic2[T_cc_otsu]; m3 = dic3[T_mcc_otsu];	m4 = dic4[T_tri_otsu]
					
					b100_len = max(min(JIdic_cum[m1],CCdic_cum[m2]),Mccdic_cum[m3],Tridic_cum[m4])
					b0_len = g_size-b100_len
					#print(j, b100_len, b0_len)
					j += 1
					
					if b0_len == 0:
						c.append(b100_len)
					else:
						c.append(abs(1-(b100_len/b0_len)))

	i = c.index(min(c))

	T_ji_otsu = prop_list[i][0]; T_cc_otsu = prop_list[i][1]; T_mcc_otsu = prop_list[i][2]; T_tri_otsu = prop_list[i][3]

	next_str = str(len(T_ji_otsu_list))+str(len(T_cc_otsu_list))+str(len(T_mcc_otsu_list))+str(len(T_tri_otsu_list))

	return T_ji_otsu, T_cc_otsu, T_mcc_otsu, T_tri_otsu, next_str

def getMaxThreshold(JIdic, CCdic, Mccdic, Tridic):
	
	#print([(k, Tridic[k]) for k in sorted(Tridic)])
	T_ji_otsu_list = computeMultiOtsuThreshold(JIdic)
	T_cc_otsu_list = computeMultiOtsuThreshold(CCdic)
	T_mcc_otsu_list = computeMultiOtsuThreshold(Mccdic)
	T_tri_otsu_list = computeMultiOtsuThreshold(Tridic)

	T_ji_otsu = max(T_ji_otsu_list)
	T_cc_otsu = max(T_cc_otsu_list)
	T_mcc_otsu = max(T_mcc_otsu_list)
	T_tri_otsu = max(T_tri_otsu_list)

	print(T_ji_otsu_list)
	print(T_cc_otsu_list)
	print(T_mcc_otsu_list)
	print(T_tri_otsu_list)

	next_str = str(len(T_ji_otsu_list))+str(len(T_cc_otsu_list))+str(len(T_mcc_otsu_list))+str(len(T_tri_otsu_list))

	return T_ji_otsu, T_cc_otsu, T_mcc_otsu, T_tri_otsu, next_str

def multiOtsuCommon2(G, edge_dic, edge_prop_dic_old):
	JIdic, CCdic, Mccdic, Tridic, B0, B100, Rest_edges, edge_prop_dic = computeParameters2(G, edge_dic)
	T_ji_otsu, T_cc_otsu, T_mcc_otsu, T_tri_otsu, next_str = getOptimumThres(G, JIdic, CCdic, Mccdic, Tridic, B0, B100)

	print(T_ji_otsu, T_cc_otsu, T_mcc_otsu, T_tri_otsu)
	print(JIdic, CCdic, Mccdic, Tridic)
	for e in Rest_edges:
		ji = edge_prop_dic_old[e][0]
		cc = edge_prop_dic_old[e][1]
		mcc = edge_prop_dic_old[e][5]
		tri_ratio = edge_prop_dic_old[e][6]
		if (ji >= T_ji_otsu/2 and cc > T_cc_otsu/2) or mcc >= T_mcc_otsu/2 or tri_ratio > T_tri_otsu/2:
		#if (cc >= T_cc_otsu/2):
			B100.append(e)
		else:
			B0.append(e)

	print(next_str)
	return B0, B100, next_str


def computeFilteredHist_multiOtsu_repeatedHeighest(G, edge_dic):
	B100_set = set([])
	G1 = nx.Graph()
	G1.add_edges_from(G.edges())
	B0, B100, edge_prop_dic_old = multiOtsuCommonBest2(G1, edge_dic)

	'''
	CM = [[0,0],[0,0]]
	computeCM2(B0, 0, CM, edge_dic)
	computeCM2(B100, 1, CM, edge_dic)

	ComputePRA2(CM)
	'''
	B100_set = B100_set.union(set(B100))
	#getDistribution(B0, edge_prop_dic_old)
	prev_str = ""
	while len(B0) > len(B100):		
		G1 = nx.Graph()
		G1.add_edges_from(B0)
		B0, B100, next_str = multiOtsuCommon2(G1, edge_dic, edge_prop_dic_old)
		if prev_str == next_str:
			break
		else:
			prev_str = next_str
		B100_set = B100_set.union(set(B100))


	B100 = list(B100_set)
	CM = [[0,0],[0,0]]
	computeCM2(B0, 0, CM, edge_dic)
	computeCM2(B100, 1, CM, edge_dic)

	ComputePRA2(CM)

	return B0, B100
	

def addWedgesInB100(original_G, B0, B100, edge_dic):
	capacity = 20000
	buffer_cap = 5000
	nbd_dic = {}

	B0 = set(B0)
	B100 = set(B100)
	
	G1 = nx.Graph()
	G1.add_edges_from(B100)
	#add wedges inside G1 - made from B100
	node_deg = dict(original_G.degree())
	Edges_to_be_add = set([])
	b100_nodes = set(G1.nodes())
	mydic = {}
	for e in B0:
		e = getEdge(e)
		'''
		if e[0] in b100_nodes and e[1] in b100_nodes:
			nbd_e0_G1 = intbitset([n1 for n1 in G1.neighbors(e[0])])
			nbd_e1_G1 = intbitset([n1 for n1 in G1.neighbors(e[1])])
			if len(nbd_e0_G1 & nbd_e1_G1) > 0:
				Edges_to_be_add.add(e)
		'''
		if e[0] in b100_nodes and e[1] in b100_nodes:
			nbd_e0_G1 = getNbd(G1, e[0], capacity, nbd_dic)
			nbd_e1_G1 = getNbd(G1, e[1], capacity, nbd_dic)
			mydic[e] = len(nbd_e0_G1 & nbd_e1_G1)

		if e[0] not in b100_nodes and node_deg[e[0]] <= 2:
			Edges_to_be_add = Edges_to_be_add.union(set([(e[0], n1) for n1 in original_G.neighbors(e[0])]))
		if e[1] not in b100_nodes and node_deg[e[1]] <= 2:
			Edges_to_be_add = Edges_to_be_add.union(set([(e[1], n1) for n1 in original_G.neighbors(e[1])]))

		#if node_deg[e[0]] <= 2 and node_deg[e[1]] <= 2:
			#Edges_to_be_add.add(e)

		#if (node_deg[e[0]] >= 14 and node_deg[e[1]] <= 4) or (node_deg[e[1]] >= 14 and node_deg[e[0]] <= 4):
			#Edges_to_be_add.add(e)
	
		
	mytup = sorted(mydic.items(), key=lambda x: x[1], reverse=True)
	max_val = mytup[0][1]
	#print(mytup)
	for tup in mytup:
		if tup[1] >= max_val:
			Edges_to_be_add.add(getEdge(tup[0]))
		else:
			break

	B0 = B0.difference(Edges_to_be_add)
	B100 = B100.union(Edges_to_be_add)

	'''
	for e in B0:
		if edge_dic[e] == no_of_perm:
			print(e)
	'''
	CM = [[0,0],[0,0]]
	computeCM2(B0, 0, CM, edge_dic)
	computeCM2(B100, 1, CM, edge_dic)

	ComputePRA2(CM)

	return B0, B100
				
def computeFilteredHist_multiOtsu_repeatedHeighest_addWedge(G, edge_dic):
	B0, B100 = computeFilteredHist_multiOtsu_repeatedHeighest(G, edge_dic)
	if len(B0) > len(B100):
		return addWedgesInB100(G, B0, B100, edge_dic)
	return B0, B100


def removeNodesFromB100(original_G, B0, B100, edge_dic):		
	capacity = 20000
	buffer_cap = 5000
	nbd_dic = {}

	B_100 = set([])
	for e in B100:
		B_100.add(getEdge(e))

	B100 = B_100
	
	B_0 = set([])
	for e in B0:
		B_0.add(getEdge(e))

	B0 = B_0

	G1 = nx.Graph()
	G1.add_edges_from(B100)


	Edges_to_be_deleted = set([])
	node_deg = dict(original_G.degree())
	node_clus_sequence = nx.clustering(original_G)
	for e in B100:
		nbd_e0_G1 = getNbd(original_G, e[0], capacity, nbd_dic)
		nbd_e1_G1 = getNbd(original_G, e[1], capacity, nbd_dic)

		inter = nbd_e0_G1 & nbd_e1_G1

		if node_deg[e[0]] == 2 and node_deg[e[1]] == 2 and len(inter) == 0:# and node_clus_sequence[e[0]] == 0 and  node_clus_sequence[e[1]] == 0:
			Edges_to_be_deleted.add(e)

		'''
		inter_list = list(inter)
		if node_deg[e[0]] >= 6 and node_deg[e[1]] >= 6 and len(inter) == 1 and node_deg[inter_list[0]] >= 6 :
			Edges_to_be_deleted.add(e)
		'''
	for n in G1.nodes():
		if node_deg[n] == 2:
			nbd = getNbd(original_G, n, capacity, nbd_dic)
			if not original_G.has_edge(nbd[0], nbd[1]) and node_deg[nbd[0]] >= 6 and node_deg[nbd[1]] >= 6:
				Edges_to_be_deleted.add(getEdge((n, nbd[0])))
				Edges_to_be_deleted.add(getEdge((n, nbd[1])))
				

	B0 = B0.union(Edges_to_be_deleted)
	B100 = B100.difference(Edges_to_be_deleted)


	
	for e in B100:
		#if edge_dic[e] < no_of_perm:
			#print(e)
		if e == (1714, 8965):
			print(edge_dic[e])
 
	CM = [[0,0],[0,0]]
	computeCM2(B0, 0, CM, edge_dic)
	computeCM2(B100, 1, CM, edge_dic)

	ComputePRA2(CM)

	return B0, B100

#def thirdRoundAddfromB0(original_G, B0, B100, edge_dic):


########################### computing the Filter class - end ###########################

########################### computing threshold - start ###########################	

def getMean(cluster, tup_Prop_freq):
	s = 0
	t = 0
	for tup in tup_Prop_freq:
		if (tup[0] >= cluster[0] and tup[0] <= cluster[1]):
			s += tup[0]*tup[1]
			t += tup[1]
		if tup[0] > cluster[1]:
			break
	if t == 0:
		return 0
	else:
		return s/t

def getBetaIndex_old(cluster, local_mean, overall_mean, tup_Prop_freq):
	s = 0
	t = 0
	for tup in tup_Prop_freq:
		if (tup[0] >= cluster[0] and tup[0] <= cluster[1]):
			s += tup[1]*(tup[0] - overall_mean)**2
			t += tup[1]*(tup[0] - local_mean)**2
		if tup[0] > cluster[1]:
			break
	return s/t

def getBetaIndex(cluster1, cluster2, mean1, mean2, tup_Prop_freq):
	s = 0
	for tup in tup_Prop_freq:
		if (tup[0] >= cluster1[0] and tup[0] <= cluster1[1]):
			s += tup[1]*(tup[0] - mean1)**2
		if (tup[0] > cluster2[0] and tup[0] <= cluster2[1]):
			s += tup[1]*(tup[0] - mean2)**2
	return s

		
def getBetaThreshold(T_list, Prop_freq_dic):
	if len(T_list) == 1:
		return T_list[0]
	tup_Prop_freq = [(k, Prop_freq_dic[k]) for k in sorted(Prop_freq_dic)]
	total_prop = sum([k*Prop_freq_dic[k] for k in Prop_freq_dic])
	total_items = sum([Prop_freq_dic[k] for k in Prop_freq_dic]) 
	overall_mean = total_prop/total_items
	beta_index_th_dic = {}
	for i in range(len(T_list)):
		cluster1 = (0, T_list[i])
		cluster2 = (T_list[i], 1)
		local_mean1 = getMean(cluster1, tup_Prop_freq)
		local_mean2 = getMean(cluster2, tup_Prop_freq)
		num = getBetaIndex(cluster1, cluster2, overall_mean, overall_mean, tup_Prop_freq)
		den = getBetaIndex(cluster1, cluster2, local_mean1, local_mean2, tup_Prop_freq)
		beta_index_th_dic[i] = num/den

	if len(beta_index_th_dic) > 0:
		max_beta_index = max(beta_index_th_dic, key=beta_index_th_dic.get)
		return T_list[max_beta_index]
	else:
		return 0

	
def getBetaThreshold_old(T_list, Prop_freq_dic):
	if len(T_list) == 1:
		return T_list[0]
	tup_Prop_freq = [(k, Prop_freq_dic[k]) for k in sorted(Prop_freq_dic)]
	total_prop = sum([k*Prop_freq_dic[k] for k in Prop_freq_dic])
	total_items = sum([Prop_freq_dic[k] for k in Prop_freq_dic]) 
	overall_mean = total_prop/total_items
	beta_index_th_dic = {}
	for i in range(len(T_list)+1):
		if i == 0:#first threshold
			cluster = (0, T_list[0])
		elif i == len(T_list):#last threshold
			cluster = (T_list[i-1], 1)
		else:
			cluster = (T_list[i-1], T_list[i])
		local_mean = getMean(cluster, tup_Prop_freq)
		beta_index = getBetaIndex(cluster, local_mean, overall_mean, tup_Prop_freq)
		beta_index_th_dic[i] = beta_index

	print(beta_index_th_dic)
	max_beta_index = max(beta_index_th_dic, key=beta_index_th_dic.get)
	if max_beta_index == 0:
		return 0
	elif max_beta_index == len(T_list):
		return 1
	else:
		return T_list[max_beta_index]


def computeMultiOtsuThreshold(Prop_freq_dic):#return the first threshold
	tup_Prop_freq = [(k, Prop_freq_dic[k]) for k in sorted(Prop_freq_dic)]
	myhist = np.array([[t[1]] for t in tup_Prop_freq])
	if len(tup_Prop_freq) == 1:
		return tup_Prop_freq[0]
	L = len(myhist)
	M = len(myhist)//2
	N = L // M

	norm_hist = otsu.normalised_histogram_binning(myhist, M=M, L=L)
	valleys = otsu.find_valleys(norm_hist)
	thresholds_index = otsu.threshold_valley_regions(myhist, valleys, N)
	thresholds = [tup_Prop_freq[i][0] for i in thresholds_index]
	sorted_thresholds = sorted(thresholds)
	if len(sorted_thresholds) > 0:
		return sorted_thresholds
	else:
		return [0]



def getPixelCountsAndAvg3(i, list_tup):#fastest implementation
	left = 0
	right = 0
	left_m = 0
	right_m = 0
	global prev_left, prev_left_m, prev_right, prev_right_m

	if(i==0):
		for t in list_tup[0:i+1]:
			left += t[1]
			left_m += t[0]*t[1]
		for t in list_tup[i+1:len(list_tup)]:
			right += t[1]
			right_m += t[0]*t[1]
	else:
		left = prev_left + list_tup[i][1]
		left_m = prev_left_m + list_tup[i][0]*list_tup[i][1]
		right = prev_right - list_tup[i][1]
		right_m = prev_right_m - list_tup[i][0]*list_tup[i][1]

	prev_left = left
	prev_left_m = left_m
	prev_right = right
	prev_right_m = right_m
		
	if left == 0 and right == 0:
		return 0,0,0,0
	elif left == 0:
		return 0, right, 0, (right_m/right)
	elif right == 0:
		return left, 0, (left_m/left), 0
	return left, right, (left_m/left), (right_m/right)


def computeOtsuThreshold2(ccdic):
	maxVarBetween = 0
	finalT = (max(ccdic) + min(ccdic))/2
	sorted_ccdic =	[(i, ccdic[i]) for i in sorted(ccdic)] #A list of tuples
	for i in range(len(sorted_ccdic)):
		nOT, nBT, miO_T, miB_T = getPixelCountsAndAvg3(i, sorted_ccdic)
		varBetween = nOT*nBT*(miO_T-miB_T)*(miO_T-miB_T)
		if maxVarBetween < varBetween:
			maxVarBetween = varBetween
			finalT = sorted_ccdic[i][0]

	return finalT

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






def computeHistConvexHull(hist):
    from numpy import ones,vstack
    from numpy.linalg import lstsq
    
    sorted_x = sorted(hist)
    selected_high_slope_x = [sorted_x[0]]
    selected_x = None
    init_index_x = 0
    while init_index_x < len(sorted_x)-1:
        #print(init_index_x)
        x0 = sorted_x[init_index_x]
        y0 = hist[x0]
        max_slope = -5555555
        for j in range(init_index_x+1, len(sorted_x)):
            x1 = sorted_x[j]
            y1 = hist[x1]
            num = (y1 - y0)
            den = (x1 - x0)
            slope = num/den
            if slope > max_slope:
                max_slope = slope
                selected_x = x1
        init_index_x = sorted_x.index(selected_x) #update the init point
        selected_high_slope_x.append(selected_x) 
        
    #find the deepest concavity points as thresholds
    init_x = selected_high_slope_x[0]
    init_y = hist[init_x]
    thresholds = []
    max_diff_all = []
    for next_x in selected_high_slope_x[1:]:
        next_y = hist[next_x]
        points = [(init_x, init_y),(next_x, next_y)]
        x_coords, y_coords = zip(*points)
        A = vstack([x_coords,ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords, rcond=None)[0]
        #get the max depth
        max_diff = 0
        selected_x_l = None
        for x_l in np.arange(init_x, next_x, 0.01):
            y_l = m*x_l + c #get the point in the convex side
            y_hist = 0 if x_l not in sorted_x else hist[x_l]
            diff = y_l - y_hist
            if diff >= max_diff:
                max_diff = diff
                selected_x_l = x_l
        max_diff_all.append(max_diff)
        thresholds.append(selected_x_l)        
        init_x = next_x
        init_y = next_y
    
    #get the index of max diff
    max_dif_index = max_diff_all.index(max(max_diff_all))
    #get the corresponding threshold
    th = thresholds[max_dif_index]
    return th
        
def computeFilteredConvex(G, edge_dic, path=None):
	JIdic, CCdic, Mccdic, Tridic, B0, B100, Rest_edges, edge_prop_dic = computeParameters2(G, edge_dic)
	#dumpProp(JIdic, CCdic, Mccdic, Tridic, B0, B100, Rest_edges, edge_prop_dic, path)
	#JIdic, CCdic, Mccdic, Tridic, B0, B100, Rest_edges, edge_prop_dic = loadProp(path)

	T_ji_otsu = computeHistConvexHull(JIdic)
	T_cc_otsu = computeHistConvexHull(CCdic)
	T_mcc_otsu = computeHistConvexHull(Mccdic)
	T_tri_otsu = computeHistConvexHull(Tridic)

	CM = [[0,0],[0,0]]
	
	for e in Rest_edges:
		e = getEdge(e)
		ji = edge_prop_dic[e][0]
		cc = edge_prop_dic[e][1]
		mcc = edge_prop_dic[e][5]
		tri_ratio = edge_prop_dic[e][6]
		if (ji >= T_ji_otsu/2 and cc > T_cc_otsu/2 ) or mcc >= T_mcc_otsu or tri_ratio > T_tri_otsu/2:
			B100.append(e)
		else:
			B0.append(e)

	computeCM2(B0, 0, CM, edge_dic)
	computeCM2(B100, 1, CM, edge_dic)

	ComputePRA2(CM)

	return B0, B100


def computeHistBalanced(hist, min_count = 5):
	hist = sorted(hist.items(), key=lambda x: x[0])
	n_bins = len(hist)
	h_s = 0
	while hist[h_s][1] < min_count:
		h_s += 1  # ignore small counts at start
	h_e = n_bins - 1
	while hist[h_e][1] < min_count:
		h_e -= 1  # ignore small counts at end

	h_c = (h_s + h_e) // 2
	w_l = np.sum([hist[i][1] for i in range(h_s, h_c)])
	w_r = np.sum([hist[i][1] for i in range(h_c, h_e + 1)])

	#print(w_l, w_r)
	while h_s < h_e:
		if w_l > w_r:  # left part became heavier
			w_l -= hist[h_s][1]
			h_s += 1
		else:  # right part became heavier
			w_r -= hist[h_e][1]
			h_e -= 1

		new_c = int(round((h_e + h_s) / 2))  # re-center the weighing scale
		if new_c < h_c:  # move bin to the other side
			w_l -= hist[h_c][1]
			w_r += hist[h_c][1]
		elif new_c > h_c:
			w_l += hist[h_c][1]
			w_r -= hist[h_c][1]

		h_c = new_c
	print(hist[h_c][0])
	return hist[h_c][0]

def computeFilteredBalanced(G, edge_dic, path=None):
	JIdic, CCdic, Mccdic, Tridic, B0, B100, Rest_edges, edge_prop_dic = computeParameters2(G, edge_dic)
	#path = 'dumpProp/dblp/'
	#dumpProp(JIdic, CCdic, Mccdic, Tridic, B0, B100, Rest_edges, edge_prop_dic, path)
	#JIdic, CCdic, Mccdic, Tridic, B0, B100, Rest_edges, edge_prop_dic = loadProp(path)

	T_ji_otsu = computeHistBalanced(JIdic)
	T_cc_otsu = computeHistBalanced(CCdic)
	T_mcc_otsu = computeHistBalanced(Mccdic)
	T_tri_otsu = computeHistBalanced(Tridic)

	CM = [[0,0],[0,0]]
	
	for e in Rest_edges:
		e = getEdge(e)
		ji = edge_prop_dic[e][0]
		cc = edge_prop_dic[e][1]
		mcc = edge_prop_dic[e][5]
		tri_ratio = edge_prop_dic[e][6]
		if (ji >= T_ji_otsu/2 and cc > T_cc_otsu/2 ) or mcc >= T_mcc_otsu or tri_ratio > T_tri_otsu/2:
			B100.append(e)
		else:
			B0.append(e)

	computeCM2(B0, 0, CM, edge_dic)
	computeCM2(B100, 1, CM, edge_dic)

	ComputePRA2(CM)

	return B0, B100

########################### fetching ground truth- end ###########################

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

