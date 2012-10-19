# -*- coding: utf-8 -*-
"""
Application of nested sampling to the Light House problem from Sivia and
Skilling 2006, p 192

"The aim of this application module is to solve the 'lighthouse' problem of
Section 2.4, using the locations of flashes observed along the coastline to
locate the lighthouse that emitted them in random directions. The lighthouse is
here assumed to be somewhere in the rectangle −2 < x < 2 , 0 < y < 2 , with
uniform prior.

	
	              u=0                                 u=1
	               -------------------------------------
	          y=2 |:::::::::::::::::::::::::::::::::::::| v=1
	              |::::::::::::::::::::::LIGHT::::::::::|
	         north|::::::::::::::::::::::HOUSE::::::::::|
	              |:::::::::::::::::::::::::::::::::::::|
	              |:::::::::::::::::::::::::::::::::::::|
	          y=0 |:::::::::::::::::::::::::::::::::::::| v=0
	 --*--------------*----*--------*-**--**--*-*-------------*--------
	             x=-2          coastline -->east      x=2
	 Problem:
	  Lighthouse at (x,y) emitted n flashes observed at D[.] on coast.

	  Position    is 2-dimensional -2 < x < 2, 0 < y < 2 with flat prior
	  Likelihood  is L(x,y) = PRODUCT[k] (y/pi) / ((D[k] - x)^2 + y^2)

	  Evidence    is Z = INTEGRAL L(x,y) Prior(x,y) dxdy
	  Posterior   is P(x,y) = L(x,y) / Z estimating lighthouse position
	  Information is H = INTEGRAL P(x,y) log(P(x,y)/Prior(x,y)) dxdy

Nested sampling proceeds according to the shape of the likelihood contours, irrespective
of the actual values. Nevertheless, it is the likelihood values that define successive
objects’ weights, which define the posterior and sum to Z, and which suggest
when a run may be terminated. When these are used, it transpires that the evidence was
ln(Z/km^{−64}) = −160.29 +/- 0.16 (as the 64 data were given in kilometres), and the
specified 1000 iterates were enough. Meanwhile, the position of the lighthouse as estimated
from the given data was x = 1.24 +/- 0.18, y = 1.00 +/- 0.19. 

The current code runs the sampling and display the x, y scatter of the sampled
points color coded by their likelihood values.
"""
from nest import NestedSampler, Sample, Model
import numpy as np

uniform = np.random.random

class LightHouseModel(Model):
	"""
	
	              u=0                                 u=1
	               -------------------------------------
	          y=2 |:::::::::::::::::::::::::::::::::::::| v=1
	              |::::::::::::::::::::::LIGHT::::::::::|
	         north|::::::::::::::::::::::HOUSE::::::::::|
	              |:::::::::::::::::::::::::::::::::::::|
	              |:::::::::::::::::::::::::::::::::::::|
	          y=0 |:::::::::::::::::::::::::::::::::::::| v=0
	 --*--------------*----*--------*-**--**--*-*-------------*--------
	             x=-2          coastline -->east      x=2
	 Problem:
	  Lighthouse at (x,y) emitted n flashes observed at D[.] on coast.

	 Inputs:
	  Prior(u)    is uniform (=1) over (0,1), mapped to x = 4*u - 2; and
	  Prior(v)    is uniform (=1) over (0,1), mapped to y = 2*v; so that
	  Position    is 2-dimensional -2 < x < 2, 0 < y < 2 with flat prior
	  Likelihood  is L(x,y) = PRODUCT[k] (y/pi) / ((D[k] - x)^2 + y^2)

	 Outputs:
	  Evidence    is Z = INTEGRAL L(x,y) Prior(x,y) dxdy
	  Posterior   is P(x,y) = L(x,y) / Z estimating lighthouse position
	  Information is H = INTEGRAL P(x,y) log(P(x,y)/Prior(x,y)) dxdy
	"""

	def __init__(self):
		D = np.array([ 4.73,  0.45, -1.73,  1.09,  2.19,  0.12,
			1.31,  1.00,  1.32,  1.07,  0.86, -0.49, -2.59,  1.73,  2.11,
			1.61,  4.98,  1.71,  2.23,-57.20,  0.96,  1.25, -1.56,  2.45,
			1.19,  2.17,-10.66,  1.91, -4.16,  1.92,  0.10,  1.98, -2.51,
			5.55, -0.47,  1.91,  0.95, -0.78, -0.84,  1.72, -0.01,  1.48,
			2.70,  1.21,  4.41, -4.79,  1.33,  0.81,  0.20,  1.58,  1.29,
			16.19,  2.75, -2.38, -1.79,  6.50,-18.53,  0.72,  0.94,  3.64,
			1.94, -0.11,  1.57,  0.57])

		Model.__init__(self, 2, D)

	def lnp(self, pos):
		logL = np.log( (pos[1] / np.pi ) / ( (self.data-pos[0])*(self.data-pos[0]) + pos[1]**2 ) )
		return logL.sum()

	def fromPrior(self):
		"""
		Draw the parameters from the prior
		"""
		Obj = Sample()
	    	#Random position between [-2,2] x [0,2]
	    	Obj.prior = uniform( len(self) ) # uniform in (0,1)
	    	Obj.pos   = np.array( [4.0, 2.0] ) * Obj.prior  - np.array( [2.0, 0.0]) 
	    	Obj.logL  = self.lnp( Obj.pos )
		return Obj

	def proposal(self, guess, step):
		# Trial object
		Try = Sample() 
		Try.prior  = guess + step * ( 2. * uniform(len(self)) - 1. )  # |move| < step
		Try.prior -= np.floor(Try.prior)                     # wraparound to stay within (0,1)
	    	Try.pos    = np.array( [4.0, 2.0] ) * Try.prior  - np.array( [2.0, 0.0]) 
		return Try

def lighthouse_main(n = 100, max_iter=2000):
	""" Run the sampling """
	#n=100                 # number of objects
	#max_iter = 2000         # number of iterations
	mod = LightHouseModel()
	guess = [ mod.fromPrior() for k in xrange(n) ]

	xx = np.array([ (rk.pos[0], rk.pos[1], rk.logL) for rk in guess])

	sampler = NestedSampler(mod) 
	sampler.run_nested(guess, max_iter)
	sampler.process_results()
	return xx, sampler, mod

def lighthouse_plot(xx, sampler, mod):
	""" Display the results """
	import pylab as plt
	plt.scatter(sampler.flatsamples[:,0], sampler.flatsamples[:,1], 
			c=np.exp(sampler.lnprobability), 
			edgecolor='None', s=20)

	plt.plot(xx[:,0], xx[:,1], 'o', mfc='None', ms=10)
	plt.show()

if __name__ == "__main__":
	xx, sampler, mod = lighthouse_main(100, 2000)
	lighthouse_plot(xx, sampler, mod)


