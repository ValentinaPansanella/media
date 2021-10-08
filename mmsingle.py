import os
import sys
import future.utils
import warnings
# warnings.filterwarnings("ignore")
from datetime import datetime
import pickle
import abc
import numpy as np
import past.builtins
import six
import tqdm
import networkx as nx
import csv
from sklearn.cluster import MeanShift
import ndlib.models.ModelConfig as mc
from ndlib.viz.mpl.OpinionEvolution import OpinionEvolution
import ndlib.models.opinions as op
from netdispatch.AGraph import AGraph


def dist(ops, i, j):
    return abs(ops[i]-ops[j])

def avg_pw_dists(ops):
    dists = 0
    for i in range(len(ops)):
        for j in range(i, len(ops)):
            dists += dist(ops, i, j)
    avgd=dists/(len(ops)**2)
    return avgd

def clustering_mean_shift(ops):
    sorted_ops = sorted(ops.items(), key = lambda kv:(kv[1], kv[0]))
    A=np.array([el[1] for el in sorted_ops]).reshape(-1,1)
    clustering = MeanShift(bandwidth=0.01).fit(A)
    lbls = clustering.labels_
    labels = np.arange(len(sorted_ops))
    for i in range(len(labels)):
        cl = lbls[i]
        labels[sorted_ops[i][0]]=cl
    return labels


def avg_nclusters(ops):   
    labels = clustering_mean_shift(ops)
    cluster_participation_dict = {}
    for l in labels:
        if l not in cluster_participation_dict:
            cluster_participation_dict[l] = 1
        else:
            cluster_participation_dict[l] += 1
    #computing effective number of clusters using function explained in the paper
    C_num = 0
    C_den = 0
    for k in cluster_participation_dict:
        C_num += cluster_participation_dict[k]
        C_den += ((cluster_participation_dict[k])**2)
    C_num = (C_num**2)
    C = C_num/C_den
    return labels, C

def Average(ops):
    return sum(ops)/len(ops)


def execution(e, p, g):
    print('e={}, p={}, g={}'.format(e, p, g))
    graph = nx.complete_graph(100)
    model = op.AlgorithmicBiasMediaModel(graph) 
    #model configuration
    config = mc.Configuration()
    config.add_model_parameter("epsilon", e)
    config.add_model_parameter("gamma", g)
    config.add_model_parameter("gamma_media", g)
    config.add_model_parameter("p", p)
    config.add_model_parameter("k", 3)
    model.set_initial_status(config)
    model.set_media_opinions([0.2, 0.5, 0.8])
    iterations = model.steady_state(max_iterations=1000000, nsteady=1000, sensibility=0.00001, node_status=True, progress_bar=True, drop_evolution=False)
    with open('C:\\Users\\valen\\Documents\\GitHub\\2019_Pansanella\\datasets\\out\\mediamodel\\e{}_p{}_g{}_iterationsdict.pickle'.format(e,p,g), 'wb') as outfile:
        pickle.dump(iterations, outfile)
        print('done')
    finalops = iterations[len(iterations)-1]['status'] #dizionario nodo->opinione all'ultima iterazione 
    ops = list(finalops.values()) #lista opinioni all'ultima iterazione
    with open('C:\\Users\\valen\\Documents\\GitHub\\2019_Pansanella\\code\\datasets\\out\\mediamodel\\e{}_p{}_g{}_finalops.ops'.format(e, p, g), 'w') as outfile:
        for o in ops:
            outfile.write(str(o)+'\n')
    with open('C:\\Users\\valen\\Documents\\GitHub\\2019_Pansanella\\datasets\\out\\mediamodel\\e{}_p{}_g{}_singleexecution.csv'.format(e,p,g), 'a+') as outfile:
        nc = avg_nclusters(finalops)
        pwd = avg_pw_dists(finalops)
        nit = iterations[len(iterations)-1]['iteration']
        outfile.write(str(nc)+','+str(pwd)+','+str(nit)+'\n')
    return 0

for p in [0.3, 0.5]:
    for e in [0.3, 0.5]:
        for g in [1.0, 1.25, 1.5]:
            res = execution(e,p,g)
            if res != 0:
                print('qualcosa è andato storto')