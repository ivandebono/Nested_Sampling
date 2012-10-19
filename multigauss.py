# -*- coding: utf-8 -*-
"""
Application of nested sampling to a triple gaussian likelihood
"""
from nest import NestedSampler, Sample, Model
import numpy as np
import pylab as plt
from figure import GridData

uniform = np.random.random

class MultiGaussModel(Model):
	def __init__(self, Ndim):
		self.D1 = 1. / np.random.rand(Ndim)
		self.D2 = 1. / np.random.rand(Ndim)
		self.D3 = 1. / np.random.rand(Ndim)
		self.M1 = 2.*np.random.rand(Ndim) - 2.
		self.M2 = 2.*np.random.rand(Ndim) 
		self.M3 = 2.*np.random.rand(Ndim) -1.
		Model.__init__(self, Ndim, None)

	def lnp(self, pos):
		f = lambda pos, M, D: np.sqrt(D.sum()) * np.exp( -0.5 * ( D * (pos - M) ** 2 ).sum() )
		return np.log( f(pos, self.M1, self.D1) + f(pos, self.M2, self.D2) + f(pos, self.M3, self.D3) )

	def fromPrior(self):
		"""
		Draw the parameters from the prior
		"""
		Obj = Sample()
	    	#Random position between [-2,2] x [0,2]
	    	Obj.prior = uniform( len(self) ) # uniform in (0,1)
	    	Obj.pos   = 4. * Obj.prior -2.
	    	Obj.logL  = self.lnp( Obj.pos )
		return Obj

	def proposal(self, guess, step):
		# Trial object
		Try = Sample() 
		Try.prior  = guess + step * ( 2. * uniform(len(self)) - 1. )  # |move| < step
		Try.prior -= np.floor(Try.prior)                     # wraparound to stay within (0,1)
	    	Try.pos    = 4.* Try.prior  - 2.
		return Try

def main(Ndim = 10, n = 200, max_iter=5000):
	""" Run the sampling """
	mod   = MultiGaussModel(Ndim)
	guess = [ mod.fromPrior() for k in xrange(n) ]

	xx = np.array([ (rk.pos[0], rk.pos[1], rk.logL) for rk in guess])

	sampler = NestedSampler(mod) 
	sampler.run_nested(guess, max_iter)
	sampler.process_results()
	return xx, sampler, mod

def plot(xx, sampler, mod):
	""" Display the results """
	#from mpl_toolkits.mplot3d import Axes3D
	ax = plt.gcf().add_subplot(111)#, projection='3d')
	
	x,y = np.mgrid[-2:2:100j, -2:2:100j]
	x = x.ravel(); y = y.ravel()
	lnp = np.empty(len(x), dtype=float)
	for k in range(len(x)): lnp[k] = mod.lnp((x[k],y[k]))
	#ax.scatter(x,y,c=lnp, edgecolor='None', alpha=0.5, s=40)

	g0 = GridData(x,y,lnp)
	im, cb = g0.imshow(alpha=0.5)
	cb.set_label('ln(P)')
	vmin, vmax = cb.get_clim()
	ax.scatter(sampler.flatsamples[:,0], sampler.flatsamples[:,1], c=sampler.lnprobability, 
			edgecolor='None', s=20, vmin=vmin, vmax=vmax)
	
	ax.plot(mod.M1[0], mod.M1[1], 'o', mec='k',   mfc='None', ms=20, mew=4)
	ax.plot(mod.M1[0], mod.M1[1], 'o', mec='1.0', mfc='None', ms=20, mew=3)
	
	ax.plot(mod.M2[0], mod.M2[1], 'o', mec='k',   mfc='None', ms=20, mew=4)
	ax.plot(mod.M2[0], mod.M2[1], 'o', mec='1.0', mfc='None', ms=20, mew=3)

	ax.plot(mod.M3[0], mod.M3[1], 'o', mec='k',   mfc='None', ms=20, mew=4)
	ax.plot(mod.M3[0], mod.M3[1], 'o', mec='1.0', mfc='None', ms=20, mew=3)

	plt.plot(xx[:,0], xx[:,1], 'o', mfc='None', ms=10)
	plt.xlim(min(x), max(x))
	plt.ylim(min(y), max(y))
	
	plt.show()

if __name__ == "__main__":
	xx, sampler, mod = main(2, 100, 2000)
	plot(xx, sampler, mod)


