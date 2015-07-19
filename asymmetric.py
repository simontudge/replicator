"""Module for the evolutionary dynamics of asymetric games"""

import scipy.integrate as I
import numpy as np
import matplotlib.pylab as plt
import numpy.random as rand

class asym():

	"""An assymmetric game is one composed of two seprate subpopulation
	x and y. The individuals in x derive fitness by playing against 
	individuals in population y. They payoff that an individual in population x
	of type i gains from playing angainst an individual of type j from popeulation
	y is given by M_ij. Likewise an individual of type i from pop y playing 
	against an individual of type i from pop x gains payoff L_ij."""

	def __init__(self, L , M, tf = 100, points = 5000 ):
		
		self.L = L
		self.M = M
		self.tf = tf
		self.points = points

		##The dimention of the model,
		##Might be able to generalise this to so that x and y can have different dimentions
		self.dim = self.L.shape[0]

		##Initial conditions, add some options here!
		self.x0 = np.array( [1/float(self.dim)]*self.dim )
		self.y0 = np.array( [1/float(self.dim)]*self.dim )

		##The state of the population, we'll bundle up x and y
		##into a single one dimentional array, as this is the
		##only input option for scipy
		self.S0 = np.empty( 2*self.dim ) 
		self.S0[:self.dim] = self.x0
		self.S0[self.dim:] = self.y0
		
		##A vector of points at which to return y
		self.t = np.linspace( 0, self.tf, self.points )

	def _dS( self, S, t  ):
		"""Change in the state due to an infitesimal increase in time, where
		s is the state of the population, and t is the time. There is actually
		no dependence on time, but scipy expects the time argument."""

		x = S[:self.dim]
		y = S[self.dim:]

		##The change in the x half
		pi = self.M.dot( y )
		pi_bar = x.dot( pi )
		dx = x*( pi - pi_bar )

		##The change in the y half
		omega = self.L.dot( x )
		omega_bar = y.dot( omega )
		dy = y*( omega - omega_bar )

		dS = np.empty( 2*self.dim )
		dS[ : self.dim ] = dx
		dS[ self.dim : ] = dy
		
		return dS

	def graph(self):

		fig = plt.figure()

		ax1 = fig.add_subplot( 2, 1, 1 )
		ax1.plot( self.x_of_t )
		ax1.set_ylim( [ 0, 1 ] )
		ax1.set_title("x")
		ax1.legend( range(self.dim), loc = 'best' )

		ax2 = fig.add_subplot( 2, 1, 2 )
		ax2.plot( self.y_of_t )
		ax2.set_ylim( [ 0, 1 ] )
		ax2.set_xlabel( "Time" )
		ax2.set_ylabel( "Density" )
		ax2.set_title("y")
		ax2.legend( range(self.dim), loc = 'best')
		
		return fig

	def go( self ):
		"""Once the class has been initisialised, this function sets the thing in motion."""
		
		##Set the integration method going
		self.S_of_t = I.odeint( self._dS, self.S0, self.t )

		##Extract x and y from the whole state
		self.x_of_t = self.S_of_t[:,:self.dim]
		self.y_of_t = self.S_of_t[:,self.dim:]

		self.fig = self.graph()
		self.fig.show()

example_use =\
"""
dim = 5
L = rand.random( ( dim, dim ) )
M = rand.random( ( dim, dim ) )

x0 = np.array( [1/float(dim)]*dim )
y0 = np.array( [1/float(dim)]*dim )

model = asym(L,M)
model.go()
"""






