import random as rnd
import networkx as nx
import infomap
from collections import defaultdict

fileName = "../TINY_100K/football_nx.txt"
g = nx.read_edgelist(fileName, nodetype=int)

info = infomap.Infomap('-s '+str(rnd.randint(0, 999999)))
for e in list(g.edges()):
        info.addLink(*e)
info.run()
c = info.getModules() #node:community

com = defaultdict(list) 
for u in c:
    com[c[u]].append(u) #community:[nodes]

print(com)
