import numpy as np

from skopt.space import Space
from skopt.sampler import Halton
from Leapfrog import Leapfrog

def CreateTrainingData(spacedim,forces,n,h,n_h = 800):
	
	space = Space(spacedim)
	h_gen = h/n_h
	
	# Compute flow map from Halton sequence to generate learnin data
	halton = Halton()
	start = halton.generate(space, n)
	start = np.array(start).transpose()
	
	final = start.copy()

	for j in range(0,n_h+1):
		final = Leapfrog(final.copy(),h_gen,forces)
		
	return start,final

