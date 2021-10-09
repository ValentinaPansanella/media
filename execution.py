import os
import sys
import warnings
from datetime import datetime
import pickle
import abc
import numpy as np
import six
import tqdm
import networkx as nx
import csv
from sklearn.cluster import MeanShift
import ndlib.models.ModelConfig as mc
from ndlib.viz.mpl.OpinionEvolution import OpinionEvolution
import ndlib.models.opinions as op
from netdispatch.AGraph import AGraph
import pickle


def dist(ops, i, j):
    return abs(ops[i]-ops[j])

def avg_pw_dists(ops):
    dists = 0
    for i in range(len(ops)):
        for j in range(i, len(ops)):
            dists += dist(ops, i, j)
    avgd=dists/(len(ops)**2)
    return avgd

# def clustering_naive(ops, thereshold=0.001):
#     i = 0
#     d = dict()
#     for el in ops:
#         d[i] = el
#         i += 1
#     sorted_ops = sorted(d.items(), key = lambda kv:(kv[1], kv[0]))
#     labels = [0 for i in range(len(ops))]
#     for i in range(len(sorted_ops)-1):
#         dist = abs(sorted_ops[i][1]-sorted_ops[i+1][1])
#         if dist < thereshold:
#             labels[sorted_ops[i+1][0]] = labels[sorted_ops[i][0]]
#         else:
#             labels[sorted_ops[i+1][0]] = labels[sorted_ops[i][0]]+1
#     return labels

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

def model_configuration(p, e, g, g_media, k):
    #model configuration
    config = mc.Configuration()
    config.add_model_parameter("epsilon", e)
    config.add_model_parameter("gamma", g)
    config.add_model_parameter("gamma_media", g_media)
    config.add_model_parameter("p", p)
    config.add_model_parameter("k", k)
    return config

def model_setting(p, e, g, g_media, k, media_op,n):
    graph = nx.complete_graph(n)
    model = op.AlgorithmicBiasMediaModel(graph) 
    config = model_configuration(p, e, g, g_media, k)
    model.set_initial_status(config)
    model.set_media_opinions(media_op)
    return model

def write_results(iterations, name, r):
    finalops = iterations[len(iterations)-1]['status'] #dizionario nodo->opinione all'ultima iterazione
    finalopsfile = open('results/final_opinions {} r{}.csv'.format(name, r), 'w+')        
    ops = list(finalops.values()) #lista opinioni all'ultima iterazione
    for o in ops:
        finalopsfile.write(str(o)+'\n')
    finalopsfile.close()

    return finalops, ops

def plot_results(iterations, model, name, r):
    viz = OpinionEvolution(model, iterations)
    viz.plot('plots/spaghetti {} r{}.png'.format(name, r))

def save_evolution(iterations, name, r):
    with  open('results/evolution {} r{}.pickle'.format(name,r), 'wb') as outfile:
        pickle.dump(iterations, outfile)

        
    

def execution(p, e, g, g_media, k, media_op, max_iterations=1000000, n=100, nruns=10, progress_bar=True, drop_evolution=False):

    name = "media mo{} p{} e{} g{} gm{} mi{} n{} nruns{}".format(media_op, p, e, g, g_media, max_iterations, n, nruns)

    print(name)

    carr = []
    darr = []
    itarr = []
    
    for r in range(nruns):

        model =  model_setting(p, e, g, g_media, k, media_op, n)

        iterations = model.steady_state(max_iterations=max_iterations, progress_bar = progress_bar, drop_evolution=drop_evolution)
        
        finalops, ops= write_results(iterations, name, r)
        plot_results(iterations, model, name, r)
        if not drop_evolution:
            save_evolution(iterations,name,r)

        avg_dist = avg_pw_dists(ops) 
        labels, n_cluster = avg_nclusters(finalops)
        n_iter = iterations[len(iterations)-1]['iteration']

        carr.append(n_cluster)
        darr.append(avg_dist)
        itarr.append(n_iter)

    #computing averages over nruns
    avg_ncluster = np.average(np.array(carr))
    avg_pwdist = np.average(np.array(darr))
    avg_iter = np.average(np.array(itarr))

    std_ncluster = np.std(np.array(carr))
    std_pwdist = np.std(np.array(darr))
    std_iter = np.std(np.array(itarr))
    
    aggregatefile = open('aggregate/aggregate_results media mo{} n{}.csv'.format(media_op, n), 'a+')

    #string of results
    s = ','.join(['complete', str(nruns), str(n), '1.0', str(p), str(e), str(g), str(g_media),str(k), str(avg_ncluster), str(avg_pwdist), str(avg_iter), str(std_ncluster), str(std_pwdist), str(std_iter)])
    s += '\n'
   
    #writing files
    aggregatefile.write(s)
    aggregatefile.flush()
    aggregatefile.close()

def clean_output_csv(media_op, n):
    #pulire file
    opfile = open('aggregate/aggregate_results media mo{} n{}.csv'.format(media_op, n), 'w+')
def write_header_csv(media_op, n):
    opfile = open('aggregate/aggregate_results media mo{} n{}.csv'.format(media_op, n), 'w+')
    opfile.write('graph,nruns,n,density,p,eps,gam,gam_media,k,avg_ncluster,avg_pwdist,avg_niter,std_ncluster,std_pwdist,std_niter')
    opfile.write('\n')
    opfile.close()

def single_execution(p, e, g, g_media, k, media_op, max_iterations=1000000, n=100, nruns=1, progress_bar=True, drop_evolution=False):
    name = "media mo{} p{} e{} g{} gm{} mi{} n{} nruns{}".format(media_op, p, e, g, g_media, max_iterations, n, nruns)
    print(name)
    for r in range(nruns):
        model =  model_setting(p, e, g, g_media, k, media_op, n)
        iterations = model.steady_state(max_iterations=max_iterations, progress_bar = progress_bar, drop_evolution=drop_evolution)
        write_results(iterations, name, r)
        plot_results(iterations, model, name, r)
        if not drop_evolution:
            save_evolution(iterations,name,r)