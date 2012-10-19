# -*- coding: utf-8 -*-
"""
Application of nested sampling to a ndimension gaussian fit
"""
from nest import NestedSampler, Sample, Model
import numpy as np

uniform = np.random.random


class GaussModel(Model):
	def __init__(self, Ndim):
		D = 1. / np.random.rand(Ndim)
		Model.__init__(self, Ndim, D)

	def lnp(self, pos):
		return -0.5 * np.sum(self.data * pos ** 2)

	def fromPrior(self):
		"""
		Draw the parameters from the prior
		"""
		Obj = Sample()
	    	#Random position between [-2,2] x [0,2]
	    	Obj.prior = uniform( len(self) ) # uniform in (0,1)
	    	Obj.pos   = 2. * Obj.prior -1.
	    	Obj.logL  = self.lnp( Obj.pos )
		return Obj

	def proposal(self, guess, step):
		# Trial object
		Try = Sample() 
		Try.prior  = guess + step * ( 2. * uniform(len(self)) - 1. )  # |move| < step
		Try.prior -= np.floor(Try.prior)                     # wraparound to stay within (0,1)
	    	Try.pos    = 2.* Try.prior  - 1.
		return Try

def gauss_main(Ndim = 10, n = 100, max_iter=5000):
	""" Run the sampling """
	mod   = GaussModel(Ndim)
	guess = [ mod.fromPrior() for k in xrange(n) ]

	xx = np.array([ (rk.pos[0], rk.pos[1], rk.pos[2], rk.logL) for rk in guess])

	sampler = NestedSampler(mod) 
	sampler.run_nested(guess, max_iter)
	sampler.process_results()
	return xx, sampler, mod

def gauss_plot(xx, sampler, mod):
	""" Display the results """
	import pylab as plt
	from mpl_toolkits.mplot3d import Axes3D
	ax = plt.gcf().add_subplot(111, projection='3d')
	ax.scatter(sampler.flatsamples[:,0], sampler.flatsamples[:,1], sampler.flatsamples[:,2],
			c=np.exp(sampler.lnprobability), 
			edgecolor='None')#, s=20)

	plt.plot(xx[:,0], xx[:,1], xx[:,2], 'o', mfc='None', ms=10)
	plt.show()

if __name__ == "__main__":
	xx, sampler, mod = gauss_main(10, 50, 2000)
	gauss_plot(xx, sampler, mod)


