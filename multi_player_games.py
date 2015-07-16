##Generalise the dynamics of the replicator equation to more than two player and two strategies.
##We first consider the less general case in which the internal configuration of strategies is not
##important. This can be represented via a payoff matrix. The top row being every possible combination
##of strategies. i.e. AAA AAB AAC ABB ABC ACC BBB BBC BCC CCC for three stratgies. We would then need
##three rows which denote the payoff agents would recieve for being in that group.
import numpy as np
import random
##The 'choose' function
from scipy.misc import comb as C
##Integration procedure
import scipy.integrate as I
import pylab as pl
import itertools
import math

class d2_game:

	"""Special non-general case in which we have 2 strategies and d players"""

	def __init__( self, game, x0 = 0.5, tf = 100, points = 100, makeGraphs = True):
		self.game = np.array( game )
		##Read the number of strategies, and group size from the matrix
		self.n, self.d = self.game.shape
		C_matrix = np.array( [ list( C(self.d-1,range(self.d)) ) for _ in xrange(self.n) ]  )
		##We perform a simple trick in that we weight each value of the payoff matrix
		##by the number of ways in which that combination can occur.
		self.M = np.multiply( C_matrix, self.game )

		##State variable x, density of A players
		self.x0 = x0
		##Vector of points in time at which we want to know x
		self.tf = tf
		self.t = np.linspace(0,self.tf,points)

		self.makeGraphs = makeGraphs

	def x_vec(self, x):
		return np.array( [ x**(self.d-1-k)*(1-x)**k for k in xrange(self.d) ] )

	def PIs(self,x):
		return np.dot( self.M, self.x_vec(x) )

	def _dx( self, x, t0 ):
		"""This is the function which the integration procedure calls"""
		piA,piB = self.PIs(x)
		return x*(1-x)*(piA-piB)

	def go(self):

		self.x = I.odeint( self._dx, self.x0, self.t )
		self.final_x = self.x[-1][0]
		if self.makeGraphs:
			pl.figure()
			pl.plot(self.t,self.x)
			pl.xlabel( 't' )
			pl.ylabel( 'x' )
			pl.show()

class nd_game:
	"""Replicator Dynamics for games in groups of n players, with d strategies."""

	def __init__(self, game, d, weighted = False,  x0 = 'uniform', tf = 100, points = 100, makeGraphs = True):

		self.game = np.array( game )
		self.d = d
		self.n = self.game.shape[0]
		#assert( self.game.shape[1] == self.L )
		self.cases = self.case_list()
		self.L = len( self.cases )
		if not weighted:
			self.W = self.weights()
			##Transform the matrix
			weight_matrix = [ self.W for _ in xrange(self.n) ]
			self.M = np.multiply( self.game, weight_matrix )
		else:
			self.M = self.game

		##Set initial densities
		if x0 == 'uniform':
			self.x0 = [1/float(self.n)]*self.n
		elif x0 == 'random':
			x = np.array( [random.random() for _ in xrange(self.n) ] )
			self.x0 = x/sum(x)
		else:
			self.x0 = x0

		##Vector of points in time at which we want to know x
		self.tf = tf
		self.t = np.linspace(0,self.tf,points)

		self.makeGraphs = makeGraphs

		self.CS = np.array( self.counts() )

	def case_list(self):
		"""Returns a list of all the cases of the different possible opponenets."""
		return list( itertools.combinations_with_replacement( range(self.n), self.d - 1 ) )

	def counts(self):
		"""A list of how many times each strategy occurs in each group possibility"""
		O = []
		for L in self.cases:
			occurances = []
			for nn in xrange(self.n):
				occurances.append( L.count(nn) )

			O.append(occurances)
		return O

	def weights(self):
		"""Returns the list of weights that the payoff matrix needs to be multiplied by."""
		return map(self.redundancy, self.cases)

	# def length(self):
	# 	"""Returns how long the matrix should be"""
	# 	return len( self.cases )

	def redundancy(self, L):
		"""In how many ways can this combination be created."""
		occurances = []
		for nn in xrange(self.n):
			occurances.append( L.count(nn) )
		facs = map( math.factorial, occurances )
		total = reduce(lambda x,y:x*y, facs)
		return math.factorial( self.d - 1 )/total

	def PI(self,i,x):
		"""Returns the fitness of inividual i"""
		##Probability that a random group will be of the type j

		##Sum over every possibility, j:
		scores = []
		for j in xrange(self.L):
			probs = [ x[k]**self.CS[ j, k ] for k in xrange(self.n) ]
			prob = np.prod(probs)
			scores.append( self.M[i,j]*prob )
		return sum(scores)

		#return sum( [ self.M[i,j] * sum( [ x[k]**self.CS[j,k] for k in xrange(self.n) if self.CS[j,k] != 0 ] ) for j in xrange(self.L)] )

	def _dx(self,x,t):
		##Fitness
		PIs = np.array( [ self.PI(i,x) for i in xrange(self.n) ] )
		PI_bar = np.dot( PIs, x )
		d = np.multiply( x, PIs - PI_bar)
		return d

	def go(self):

		self.x = I.odeint( self._dx, self.x0, self.t )
		self.final_x = self.x[-1]
		if self.makeGraphs:
			pl.plot(self.x)
			pl.show()


def random_multi_game(d,n = 2):
	length = len( list( itertools.combinations_with_replacement( range(n), d - 1 ) ) )
	return np.array( [ [ random.random() for i in xrange( length ) ] for j in xrange(n)] )

def ST_space(p=100):
	"""Draw ST space, thi is used as a check to show that two player two strategy games are a special case of this model"""

	data = np.zeros( ( p, p ) )
	Ss = np.linspace( -1 , 1 , p )
	Ts = np.linspace(0,2,p)
	for i,S in enumerate(Ss):
		for j,T in enumerate(Ts):

			M = np.array( [ [1, S], [T,0] ] )

			model = agent_model(M, d = 2, makeGraphs = False)
			model.go()

			data[i,j] = model.final_x[0]

	pl.imshow(data,interpolation = 'nearest', origin = [ Ts[0], Ss[0] ], extent = [Ts[0],Ts[-1],Ss[0],Ss[-1]])
	pl.xlabel('T')
	pl.ylabel('S')
	pl.colorbar()
	pl.show()

def N_person_SD(model_type = 'nd'):

	##Lets have a go at replicating the results from the paper: "cooperative behaviour in a model of evolutionary snowdrfit games with N
	# person interaction. Zheng et. al. 2007"

	##The matrix from the paper:
	#cost
	c = 1
	if model_type == 'agent':
		points = 12
	else:
		points = 100
	Rs = np.linspace(1e-10,1,points)

	colors = [ 'red','blue','green','yellow']
	for i,d in enumerate( [2,3,5,10] ):
		fxs = []

		for r in Rs:
			if r == 0:
				c = 0
				b = 1
			else:
				c = 1
				b = c/r
	 
			M = np.zeros( (2,d) )
			M[0,:] = [ b - c/float( d - k) for k in xrange(d) ]
			M[1,:] = [b]*d
			M[1,-1] = 0

			if model_type == 'nd':
				model = nd_game(M, d= d, makeGraphs = False, tf = 50)
			elif model_type == 'agent':
				model = agent_model(M, pop_size = 250, d = d, makeGraphs = False)
			else:
				model = d2_game(M,makeGraphs = False, tf = 50)

			model.go()
			fxs.append( model.final_x )

		if model_type == 'agent':
			pl.plot( Rs, fxs, 'o' , label = "%d"%d, color = colors[i])
		else:
			pl.plot( Rs, fxs, color = colors[i])

	pl.ylim([-0.05,1.05])
	pl.xlim([-0.05,1.05])

	pl.legend()
	#pl.show()

def FPS(weights):
	"""From a list of fitnesses returns a list of indices, the same length as the list."""

	##Make a list of runing totals, only need to do this once.
	totals = []
	running_total = 0

	for w in weights:
		running_total += w
		totals.append(running_total)

    ##Now pick from this list multiple times
	inds = []
	for _ in xrange( len(weights) ):
		rnd = random.random() * running_total
		for i, total in enumerate(totals):
			if rnd < total:
				inds.append(i)
				break

	return inds

class agent_model:

	"""Implement a version of the model using a finite population of agents."""

	def __init__(self, M, d, pop_size = 100, gens = 1000, makeGraphs = True):

		self.d = d
		self.pop_size = pop_size - pop_size%self.d
		self.M = np.array(M)
		##Negative fitnesses are nosensical, so we scale the matrix such that its smalest value is zero
		if self.M.min() < 0:
			self.M -= self.M.min()
		##Also if fitnesses are zero then we can be in trouble
		if self.M.min() == 0:
			self.M += 1e-5
		self.n = self.M.shape[0]

		##Initialise a population
		self.population = np.array( [ random.randint( 0, self.n-1 ) for _ in xrange( self.pop_size ) ] )

		##A list of all the posible cases for the ther group members
		self.cases = list( itertools.combinations_with_replacement( range(self.n), self.d - 1 ) )

		self.make_dictionary()

		self.gens = gens

		##For recording the frequecny of every strategy
		self.y = np.zeros( (self.gens, self.n) )

		self.makeGraphs = makeGraphs

	def sort_population(self):
		"""Sorts the population into groups of size d"""
		self.sorted_pop = self.population.reshape( (self.pop_size/self.d,self.d ) )

	def score_group(self, group):
		"""Returns a list of scores for a certain group."""
		f = np.zeros( self.d )
		for i in xrange(self.d):
			##Make a group of everyone except the focal individual
			key = tuple( [ group[j] for j in xrange( self.d ) if j != i ] )
			j_ind = self.case_dic[key]
			##The i indices is simply the strategy of the focal individual
			i_ind = group[i]
			f[i] = self.M[ i_ind, j_ind ]
		
		return f

	def score_pop(self):
		"""Sets an array with the fitness of all individuals"""
		self.fits = np.array( map(self.score_group, self.sorted_pop) ).flatten()


	def make_dictionary( self ):
		"""Makes a dictionary with all the possible permuations of the possible other group mebers as the keys,
		the items are them where to look up this value in the payoff matrix."""
		self.case_dic = {}
		for i,case in enumerate( self.cases ):
			for p in itertools.permutations( case ):
				self.case_dic[ p ] = i

	def next_gen(self):
		"""Creates the next generation."""
		
		##Sort the population into groups
		self.sort_population()
		##Make an array of fitnesses
		self.score_pop()
		##Call the fitness proportionate selection function
		new_inds = FPS(self.fits)
		new_pop = []
		for ind in new_inds:
			new_pop.append( self.population[ind] )

		self.population = np.array( new_pop )

	def strat_freqs(self):
		"""Return the frequency of each strategy in the population."""
		numbers = np.zeros( self.n )
		for nn in xrange(self.n):
			numbers[ nn ] = list( self.population ).count(nn)
		return numbers/float( self.pop_size )

	def go(self):

		self.fit_time = []
		for gen in xrange( self.gens ):
			self.next_gen()
			self.y[gen,:] = self.strat_freqs()
			self.fit_time.append( np.mean( AM.fits ) )

		##Time averaged mean, obnly over the last half of the run
		self.time_mean_y = np.mean( self.y[1000/2:,:],0 )

		##This is for compatability with other modules
		self.final_x = self.time_mean_y
		if len(self.final_x) == 2:
			self.final_x = self.final_x[0]

		if self.makeGraphs:
			pl.figure()
			pl.plot(self.y)
			if self.n < 8:
				pl.legend( map( str,range(self.n) ) )
			for tmy in self.time_mean_y:
				pl.plot( [tmy]*self.gens, '--' )
			#pl.show()

def max_fit( M, d ):
	"""For a given game returns the state which maximises average fitness."""
	M = np.array(M)
	n = M.shape[0]
	
	all_cases = list( itertools.combinations_with_replacement( range(n), d ) )

	total_states = len( all_cases )

	##A cheeky hack, we'll create an instance of the agent class in oder to use its function
	#
	##CATION, REMEBER THE AGENT CLASS TRANSFORMS THE MATRIX SO THAT NO ELEMENT IS LESS THAN ZERO
	#
	agent = agent_model(M,d)
	scores = map( agent.score_group, all_cases )
	mean_scores = map(np.mean,scores)

	import operator
	max_index, max_score= max(enumerate(mean_scores), key=operator.itemgetter(1))
	optimal_group = all_cases[ max_index ]

	return max_score, optimal_group

class group_evolver():
	"""This is a GA in which the group is a unit of selection, rather than the individuals of the group. The fitness of the group is determined by the mean fitness of the
	constituents. This class inherits agent_class."""

	def __init__( self, M, d , pop_size = 250, gens = 1000, mu = 0 ):

		self.M = np.array( M )
		self.d = d
		self.n = M.shape[0]
		self.pop_size = pop_size
		self.gens = gens
		self.mu = mu

		self.AM = agent_model(M,d)

		self.cases = list( itertools.combinations_with_replacement( range(self.n), self.d ) )

		self.population = [ self.random_group() for _ in xrange(pop_size) ]


	def random_group(self):
		return [ random.randint(0,self.n-1) for _ in xrange(d) ]

	def mutate(self,group):

		nG = group

		for i in xrange(self.d):

			if random.random() < self.mu:
				nG[i] = random.randint( 0, self.n - 1 )

		return nG

	def fitness(self, group):
		##Use the agent classes function which scores every member of a group and return its mean
		return np.mean( self.AM.score_group(group) )

	def next_gen(self, fits):

		new_inds = FPS(fits)
		new_pop = []
		for ind in new_inds:
			new_pop.append( self.mutate( self.population[ind] ) )
		return new_pop

	def go(self):

		fit_time = []
		for gen in xrange(self.gens):
			fits = map( self.fitness, self.population )
			self.population = self.next_gen(fits)
			fit_time.append( np.mean(fits) )

		from scipy.stats import mode
		winner = mode(self.population)[0]
		print "Winner: ",winner

		pl.plot( fit_time )


if __name__ == '__main__':

	"""These are just some random experients"""

	# d = 3
	# c = 1
	# b = 1.5
	# M = np.zeros( (2,d) )
	# M[0,:] = [ b - c/float( d - k) for k in xrange(d) ]
	# M[1,:] = [b]*d
	# M[1,-1] = 0

	# model = agent_model(M, d= d, makeGraphs = True)
	# model.go()
	# pl.show()

	# S,T = 0.5,1.75
	# M = [ [1, S], [T,0] ]
	# ESS = S/( S + T - 1)
	# model = agent_model(M, d = 2, makeGraphs = True)
	# model.go()
	# pl.plot([ESS]*model.gens, '--' ) 
	# pl.show()

	# N_person_SD( 'sfasd' )

	# N_person_SD( 'agent' )

	# pl.show()

	# ST_space(15)

	# A = np.array( [ [ len( list( itertools.combinations_with_replacement(range(d-1),n) ) ) for d in range(1,10) ] for n in range(1,10) ] )

	# B = np.array( [ [ q(d,n) for d in range(1,10) ] for n in range(1,10) ] )

	##This is the example given in the paper: "Gokhale and Traulsen, 2009, Evolutionary games in the multiverse"
	#Groups of 4 with three possible strategies.
	# # n = 3
	# ##There Matrix is unweighted
	# M = np.array([ [-9.30, 3.83, 3.86, -1.03, -1.00, -0.96, 0.10, 0.33, 0.16, 0.20],
	#  				[0.10, -1.03, 0.13, 3.83, -1.00, 0.16, -9.30, 4.06, -0.96, 0.20],
	#  				[0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.20, 0.00, 0.00] ] )

	d,n = 5,5
	M = random_multi_game( d, n )

	GE = group_evolver(M,d)

	MS,OG = max_fit(M,d)

	GE.go()
	pl.plot( [MS]*GE.gens,'--' )

	AM = agent_model(M,d, gens = 1000, makeGraphs = False)
	AM.go()
	ft = AM.fit_time 

	pl.plot(ft,color = 'red')
	pl.legend( ['Group','Max','Individual'] )
	pl.show()

	# ESSs = []
	# reps = 100
	# for _ in xrange(reps):
	# 	model = nd_game(M,d, x0 ='random', weighted = False, makeGraphs = False)
	# 	model.go()
	# 	FX = model.final_x
	# 	for ESS in ESSs:
	# 		if np.allclose( FX, ESS, atol = 0.05 ):
	# 			break
	# 	else:
	# 		print "found a new one:"
	# 		print FX
	# 		ESSs.append(FX)

	# ##Make a simplex plot

	# import simplex
	# simplex.plotSimplex( np.array( ESSs ) )
	# pl.show()
