import execution as exe
import mediaplots as plot
import os

n=100
max_it=100000
nruns=1
media_op = [0.05, 0.5, 0.95]
plist = [0.1,0.3,0.5]
elist = [0.2, 0.3, 0.5]
glist = [0.0, 0.5, 0.75, 1.0, 1.25, 1.5]
# for p in plist:
# 	for e in elist:
# 		for g in glist:
# 			exe.execution(p, e, g, g, len(media_op), media_op, max_iterations=max_it, n=n, nruns=nruns, progress_bar=True, drop_evolution=False)

for p in plist:
	for e in elist:
		for g in glist:
			name = "media mo{} p{} e{} g{} gm{} mi{} n{} nruns{}".format(media_op, p, e, g, g, max_it, n, nruns)
			if not os.path.exists('plots/spaghetti {} r{}.csv'.format(name, nruns)):
				exe.single_execution(p, e, g, g, len(media_op), media_op, max_iterations=max_it, n=100, nruns=1, progress_bar=True, drop_evolution=False)
			else:
				continue

