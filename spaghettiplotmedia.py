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
# import matplotlib as mpl
# mpl.rcParams['agg.path.chunksize'] = 10000


rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

def spaghetti(e,p,g, quale=None):
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
		model.set_media_opinions([0.2, 0.5, 0.8])
		itdict = open('C:\\Users\\valen\\Documents\\GitHub\\2019_Pansanella\\datasets\\out\\mediamodel\\plots\\{}\\e{}_p{}_g{}_iterationsdict_{}.pickle'.format(quale,e,p,g, quale), 'rb')
	except FileNotFoundError as e:
		print('iterations dictionary file not found')
	iterations = pickle.load(itdict)
	ops = list(iterations[len(iterations)-1]['status'].values())
	filename = 'C:\\Users\\valen\\Documents\\GitHub\\2019_Pansanella\\datasets\\out\\mediamodel\\plots\\{}\\e{}_p{}_g{}_finalops_{}.ops'.format(quale,e, p, g, quale)
	if not Path(filename).is_file():
		finalopsfile = open(filename, 'w')
		for o in ops:
			s = str(o)+'\n'
			finalopsfile.write(s)
		finalopsfile.close()
	filename="C:\\Users\\valen\\Documents\\GitHub\\2019_Pansanella\\datasets\\out\\mediamodel\\plots\\{}\\opinion_ev_e{}_p{}_g{}_maxit{}_{}.png".format(quale, e, p, g, 10000000, quale)
	# if not Path(filename).is_file():
	viz = OpinionEvolution(model, iterations)
	viz.plot(filename)
	nc=viz.get_nc()
	plt.close()
	return nc

def plotfinaldist(e,p,g, nc, quale=None):
	try:
		ops=[]
		finalopsfile = open('C:\\Users\\valen\\Documents\\GitHub\\2019_Pansanella\\datasets\\out\\mediamodel\\plots\\{}\\e{}_p{}_g{}_finalops_{}.ops'.format(quale,e, p, g, quale), 'r')
		for el in finalopsfile.readlines():
			ops.append(float(el))
		finalopsfile.close()
	except FileNotFoundError:
		print('finalopsfile not found')
	sops = sorted(ops)
	nods = [i for i in range(100)]
	figsize =(6, 3)
	fig, ax = plt.subplots(figsize=figsize, dpi=600)        
	ax.scatter(x=np.array(nods), y=np.array(sops))
	ax.set_title('Final opinion distribution; C:{}'.format(nc), fontsize=10)
	ax.set_xlabel("Nodes", fontsize=10)
	ax.set_ylabel("Opinion", fontsize=10)
	ax.set_ylim(-0.1, 1.1)
	ax.yaxis.set_minor_locator(MultipleLocator(0.1))
	ax.tick_params(axis='both', which='major', labelsize=10, pad=8)                
	plt.grid(axis = 'both', which='both')
	plt.tight_layout()
	filename = 'C:\\Users\\valen\\Documents\\GitHub\\2019_Pansanella\\datasets\\out\\mediamodel\\plots\\{}\\e{}_p{}_g{}_finalops_{}.png'.format(quale,e,p,g, quale)
	# if not Path(filename).is_file():
	plt.savefig(filename)
	plt.close()

for e in [0.5]:
	for p in [0.5]:
		for g in [0.0, 0.5, 0.75, 1.0, 1.5]:
			try: 
				nc=spaghetti(e,p,g, quale='moreextreme')
				plotfinaldist(e,p,g, nc, quale='moreextreme')
				nc2=spaghetti(e,p,g, quale='onemedia')
				plotfinaldist(e,p,g, nc2, quale='onemedia')
			except Exception:
				continue
