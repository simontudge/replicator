##A class which takes as it's input a game and integrates the replicator equation

import scipy.integrate as I
import numpy as np
import pylab as pl
import random

def alpha_sweep( G, a0 = 0, af = 1, steps = 11 ):
	"""Takes a game and plots the fitness for varying degrees of alpha"""
	
	alphas = np.linspace( a0, af, steps )
	fs = []
	for a in alphas:
		R = replicator( G, makeGraphs = False, printInfo = False, alpha = a )
		fs.append( R.finalFit )

	max_diag = max( np.diag(G) )
	pl.plot(alphas,fs,'o')
	pl.plot(alphas, len(alphas)*[ max_diag ] )
	pl.xlabel('Alpha')
	pl.ylabel('Payoff')
	pl.show()

class Replicator:

	def __init__( self, game, tf = 'Default', points = 100, y0 = 'uniform', makeGraphs = True, alpha = 0, printInfo = True, long_run = False ):
		"""A game given by a 2d numpy array 'game', along with the final stopping time for the integrator 'tf
		(default = 100).

		Points: number of points to draw. (Default = 100).

		makeGraphs: boolean. Output a plot of the evolution through time using pyplot?

		Y0: can be 'uniform' or 'random' or an array of your choice.

		alpha: a measure of how assorted the population is.

		If long_run is set to true, default False, then the number of generations is multiplied by 10,
		do this to be more certain of convergence. Will have no effect if you specify tf manually.
		"""

		self.game = np.array( game )
		self.alpha = alpha
		self.makeGraphs = makeGraphs
		self.printInfo = printInfo
		self.dim = len(game)
		self.long_run = long_run

		if y0 == 'uniform':
			self.y0 = [1/float(self.dim)]*self.dim
		elif y0 == 'random':
			Y = np.array( [random.random() for _ in xrange(self.dim) ] )
			self.y0 = Y/sum(Y)
		else:
			self.y0 = y0

		##Set the default end time, or use the input value
		if tf == 'Default':
			self.tf = 50*self.dim
			if long_run:
				self.tf *= 10
		else:
			self.tf = tf
		##A vector of points at which to return y
		self.t = np.linspace(0,self.tf,points)

		self.game = self._assort( self.game, self.alpha )

		self._go()

	def _assort(self,G,alpha):
		"""Assorts a matrix by an amount alpha"""
		length = len(G)
		G_New = np.zeros( (length,length) )
		for i in xrange( length ):
			for j in xrange( length ):
				if i == j:
					G_New[i,j] = G[i,j]
				else:
					G_New[i,j] = alpha*G[i,i] + (1-alpha)*G[i,j]
		return G_New

	##Define the function for the replicator equation
	def _dy(self,y,t0):
		##Array of fitness
		f = np.dot(self.game, y)
		#print f
		##Average fitness
		phi = np.dot( y, f )
		#print phi
		##The replicator equation
		d = np.multiply( y, f - phi)
		return d

	def _go(self):
		"""Set the thing in motion"""
		self.y = I.odeint( self._dy, self.y0, self.t )

		##Record some metrics
		self.finalState = self.y[-1]
		##Fitness at the end
		f = np.dot(self.game, self.finalState)
		self.finalFit = np.dot( self.finalState, f )

		if self.printInfo:
			print "Final Fitness:",self.finalFit
			print "Final State:"
			for i,rho in enumerate(self.finalState):
				print i,rho

		##Make the graph
		if self.makeGraphs:
			pl.figure()
			#pl.subplot(212)
			for i in xrange(self.dim):
				pl.plot(self.t,self.y[:,i], label = str(i) )
				if self.dim <= 8:
					pl.legend( loc = 2 )
			pl.xlabel('t')
			pl.ylabel('rho')
			pl.show()

class mixed_replicator2D():
	"""Builds on the previous class, but now implements the thing with mixed strategies. Firstly for 2D cases, but later I'll try to generalise this."""

	def __init__( self, game, gridStep = 11, *args, **kwargs ):

		self.game = np.array( game )
		self.dim = len( self.game )
		self.gridStep = gridStep

		self.stratgyNumber = self.gridStep**(self.dim - 1)

		self.P = np.linspace( 0, 1, self.gridStep )

		self.makeExtendedGame()

		self.R = replicator( self.EG, *args, **kwargs )

		self.finalState = self.R.finalState

		#self.makeGraph()

		self.finalFit = self.R.finalFit
		
	def makeGraph(self):
		
		pl.figure()
		pl.plot(self.P,self.Y,'o')
		pl.show()

	def payoff(self,s1,s2):
		"""The payoff that strategy S1 recieves against S2, where S1 and S2 are vectors"""
		return s1*s2*self.game[1,1] + s1*( 1 - s2)*self.game[1,0] + ( 1 - s1 )*s2*self.game[0,1] + ( 1 - s1 )*( 1 - s2 )*self.game[0,0]

	def makeExtendedGame(self):
		"""Takes the base form game and makes the extened form game"""
		self.EG = np.array( [ [ self.payoff(y,x) for x in self.P ] for y in self.P ] )
		#print self.EG

class mixed_replicator():
	"""Takes an arbitrary game and finds fixed points in terms of mixed strategies. Does this by randomly sampling
	the strategy space rather than systematically."""

	def __init__(self, game, strategies = 250, *args, **kwargs ):
		
		self.game = np.array(game)
		self.dim = len(game)
		self.strategies = strategies

		self.args = args
		self.kwargs = kwargs

		self.go()


	def randomStrategy(self):
		"""Returns a random strategy"""

		S = np.array( [random.random() for _ in xrange( self.dim ) ] )
		return S/sum(S)

	def payoff(self,s1,s2):

		"""The payoff s1 recieves by playing s2"""
		score = 0
		for i,p1 in enumerate(s1):
			for j,p2 in enumerate(s2):
				##Probability that layer one plays a strategy i times probability that player 2 plays a strategy j times the payoff
				##recieved by player 1 in this situation
				score += p1*p2*self.game[i,j]
		return score

		##OR [ s1[i]*s2[j]*game[i,j] for i ... ]

	def go( self ):

		##Make a vector of ranodm strategies
		self.SVec = [ self.randomStrategy() for _ in xrange( self.strategies ) ]

		##Now make an extended payoff matrix
		self.EG = np.array( [ [ self.payoff( s1, s2 ) for s1 in self.SVec ]\
		 for s2 in self.SVec ] )

		##Now we send the extended game to the replicator class
		self.R = replicator( self.EG, *self.args, **self.kwargs )

		self.finalState = self.R.finalState

		#self.makeGraph()

		self.finalFit = self.R.finalFit

def allESS(G, printOut = True, trials = 100, alpha = 0, mixed = False):
	"""Takes a game and tries to find ESSs and the frequency with which they are reached"""

	if printOut:
		print "Game:"
		print G

	##Keep a dictionary of all ESSs
	ESSs = []
	freqs = []
	fits = []
	count = 0
	##Run the replicator many times from random initial conditions
	for _ in xrange(trials):
		if not mixed:
			R = replicator(G, y0 = 'random', makeGraphs = False, printInfo = False, alpha = alpha)
		else:
			R = mixed_replicator2D(G, y0 = 'random', makeGraphs = False, printInfo = False, alpha = alpha)
		ESS = R.finalState
		##Look to see if we already have the ESS
		for i,E in enumerate(ESSs):
			if np.allclose( ESS,E ):
				freqs[i] += 1
				break
		else:
			ESSs.append(ESS)
			freqs.append(1)
			fits.append(R.finalFit)

	freqs = np.array( freqs ) /float( trials )

	totalESSs = len(ESSs)

	if printOut:
		print ""
		print "Found", totalESSs, "ESSs"
		print ""

	d = {}
	for i in xrange(totalESSs):

		##Create a dictionary for this ESS
		info = {'State':ESSs[i], 'payoff': fits[i], 'Frequency': freqs[i] }

		##And create a dictionary of dictionaries for all ESSs
		d[i] = info

		if printOut:
			print ""
			print "ESS ",i,":"
			print ESSs[i]
			print "Frequency:"
			print freqs[i]
			print "Payoff:"
			print fits[i]
			print ""

	return d

def compare(G, mixed=False, printOut = True):
	"""Takes a game as an input and looks at the average fitness reached for no assortment, and for total assortment"""

	##Find the mean fitness under well mixed conditions
	infoWM = allESS(G, printOut = False, mixed = mixed)
	fWM = 0
	for ESS in infoWM.itervalues():
		fWM += ESS['Frequency']*ESS['payoff']

	##Find the fitness under assorted conditions
	infoA = allESS(G, printOut = False, alpha = 1, mixed = mixed)
	fA = 0
	for ESS in infoA.itervalues():
		fA += ESS['Frequency']*ESS['payoff']

	if printOut:
		print "Fitness of the well-mixed game: ", fWM
		print "Fitness of the  assorted  game: ", fA

	return fWM,fA

def allSituations(G):
	"""Gives the payoff of the well mixed game, the assorted game, and the assorted mixed stratey game"""
	f1,f2 = compare(G, printOut = False)
	_,f3 = compare(G, mixed = True, printOut = False)
	return [f1,f2,f3]

# G = [[1,0.2, 0.8], [.5, .5,2], [0,1.2,3]]

# MR = mixed_replicator( G, makeGraphs = False, printInfo = False )
# print MR.finalFit

# # ##A stag hunt like game
# r = 1
# d = 0.9
# G = np.array( [ [ 1, 1-r+d ],[ 1+r+d , 0 ] ] )

# print allSituations(G)
