##Give a few example of some of the things which can be done with the replicator code

import pylab as pl
import numpy as np
import random

import replicator

def ST_Sweep( points = 25, alpha = 0 ):
	"""Finds the ESS of the standard cooperative dilemma game for every point in ST space. And makes a graph."""
	data = np.zeros( (points,points) )

	Ss = np.linspace( -1,4,points )
	Ts = np.linspace( 0,5,points )

	##Sweep thourgh ST space
	for i,S in enumerate( Ss ):
		for j,T in enumerate( Ts ):

			G = np.array([[1,S],[T,0]])
			R = replicator.replicator(G, makeGraphs = False, printInfo = False, alpha = alpha)
			##Level of cooperation
			c = R.finalState[0]
			f = R.finalFit
			data[i,j] = f

	pl.imshow( data, vmin = 0, vmax = 2.5, origin = 'lower', interpolation = 'nearest', extent = [ Ss[0],Ss[-1],Ts[0],Ts[-1] ] )
	pl.xlabel('S')
	pl.ylabel('T')
	pl.colorbar()

def randomGame( dim = 2 ):
	"""Analyses a random game with a give dimention"""
	G = np.array( [ [ random.random() for _i in xrange(dim) ] for _j in xrange(dim) ] )
	print "Game:",G
	R = replicator.replicator( G )

##Draw total cooperation at ESS for ST space
pl.figure()
ST_Sweep()

##Analyse a random N=5 dimentional game
pl.figure()
randomGame(5)

##Plot ST space under varing degress of assortment
pl.figure()
for i,a in enumerate( np.linspace(0,.8,9) ):
	pl.subplot( 3, 3, i + 1 )
	ST_Sweep(alpha = a)

pl.show()
