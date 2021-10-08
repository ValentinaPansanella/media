import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import networkx as nx
import ndlib.models.ModelConfig as mc
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import os
import pickle
import sys
import future.utils
import matplotlib as mpl
import matplotlib.pyplot as plt 
from matplotlib import rc
mpl.rcParams["text.usetex"] = True
plt.rc('font',**{'family':'serif', 'size' : 22, 'serif':['Computer Modern Roman']})
import os
import sys
import future.utils
import warnings
# warnings.filterwarnings("ignore")
from datetime import datetime
from pathlib import Path
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
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

sns.set_style("whitegrid")

class OpinionEvolution(object):
    
    def __init__(self, model, trends):
        """
        :param model: The model object
        :param trends: The computed simulation trends
        """
        self.model = model
        self.srev = trends
        self.ylabel = "Opinion"
    
    def plot(self, fig, ax, filename=None):
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
        """
        Generates the plot

        :param filename: Output filename
        :param percentile: The percentile for the trend variance area
        """
        ops = self.srev[len(self.srev)-1]['status']
        labels, nclusters = avg_nclusters(ops)
        params = ['p', 'k', 'gamma', 'gamma_media', 'epsilon']
        paramsnames = [r'$p$', r'$k$', r'$\gamma$', r'$\gamma_{media}$', r'$\epsilon$']
        mapping = dict(zip(params, paramsnames))
        descr = ""
        infos = self.model.get_info()
        infos = infos.items()
        print(infos)
        infos = list(infos)


        for t in infos:
            descr += r"%s: $%s$, " % (mapping[t[0]], t[1])
        descr = descr[:-2].replace("_", " ")

        nodes2opinions = {}
        node2col = {}

        last_it = self.srev[-1]['iteration'] + 1
        last_seen = {}

        for it in self.srev:
            sts = it['status']
            its = it['iteration']
            for n, v in sts.items():
                if n in nodes2opinions:
                    last_id = last_seen[n]
                    last_value = nodes2opinions[n][last_id]

                    for i in range(last_id, its):
                        nodes2opinions[n][i] = last_value

                    nodes2opinions[n][its] = v
                    last_seen[n] = its
                else:
                    nodes2opinions[n] = [0]*last_it
                    nodes2opinions[n][its] = v
                    last_seen[n] = 0
                    if v < 0.4:
                        node2col[n] = '#3776ab'
                    elif 0.4 <= v <= 0.6:
                        node2col[n] = '#FFA500'
                    else:
                        node2col[n] = '#FF0000'       
        mx = 0
        for k, l in future.utils.iteritems(nodes2opinions):
            if mx < last_seen[k]:
                mx = last_seen[k]
            x = list(range(0, last_seen[k]))
            y = l[0:last_seen[k]]
            ax.plot(x, y, lw=1, alpha=0.5, color=node2col[k])

        descr = descr + r', $C: {:.5f}$'.format(nclusters)

#         ax.set_title(descr, fontsize=10)
#         ax.set_xlabel(r"$Iterations$", fontsize=10)
#         ax.set_ylabel(r"${}$".format(self.ylabel), fontsize=10)
#         ax.legend(loc="best", fontsize=10)
        ax.set_ylim(-0.1, 1.1)
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.tick_params(axis='both', which='major', labelsize=3, pad=2)                
        plt.grid(axis = 'both', which='both')
        plt.tight_layout()

def spaghetti(e,p,g,fig,ax,quale=None):
    try:
        graph = nx.complete_graph(100)
        model = op.AlgorithmicBiasMediaModel(graph) 
        config = mc.Configuration()
        config.add_model_parameter("p", p)
        config.add_model_parameter("k", 3)
        config.add_model_parameter("epsilon", e)
        config.add_model_parameter("gamma", g)
        config.add_model_parameter("gamma_media",g)
        model.set_initial_status(config)
        model.set_media_opinions([0.05, 0.5, 0.95])
        itdict = open('C:\\Users\\valen\\Documents\\GitHub\\2019_Pansanella\\datasets\\out\\mediamodel\\plots\\{}\\e{}_p{}_g{}_iterationsdict_{}.pickle'.format(quale,e,p,g, quale), 'rb')
    except FileNotFoundError as e:
        print('iterations dictionary file not found')
    iterations = pickle.load(itdict)
    ops = list(iterations[len(iterations)-1]['status'].values())
    viz = OpinionEvolution(model, iterations)
    viz.plot(fig, ax)

def spaghettigrid(p, quale=None):
    filename="spaghetti_p{}_{}.png".format(p, quale)
    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(6, 3), dpi=600, sharey=True)
    row=0
    for e in [0.2, 0.3, 0.5]:
        col=0
        for g in [0.0, 0.5, 0.75, 1.0, 1.5]: 
            spaghetti(e,p,g,fig,ax=axes[row,col],quale=quale)
            col+=1
        row+=1
    # plt.savefig("C:\\Users\\valen\\Documents\\GitHub\\2019_Pansanella\\datasets\\out\\mediamodel\\plots\\{}\\{}".format(quale, filename))
    plt.savefig("C:\\Users\\valen\\Downloads\\{}".format(filename))

mpl.rcParams["text.usetex"] = True
plt.rc('font',weight='bold',**{'family':'serif', 'size':6, 'serif':['Computer Modern Roman']})
for p in [0.1, 0.3, 0.5]:
    spaghettigrid(p=p, quale='moreextreme')
    