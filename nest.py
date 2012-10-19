# -*- coding: utf-8 -*-
"""
#                   NESTED SAMPLING MAIN PROGRAM
# (GNU General Public License software, (C) Sivia and Skilling 2006)


Although small, the following main code incorporates the main ideas and should
suffice for many applications. It is protected against over/underflow of
exponential quantities such as the likelihood L∗ by storing them as logarithms
(as in the variable logLstar), adding those values through the PLUS macro, and
multiplying them through summation. Rather than attempting greater
sophistication, the program uses the simple proclamation of steady compression
by log(t) = −1/n each step. The corresponding step-widths h are also stored as
logarithms, in logwidth.

The new technique of nested sampling (Skilling 2004) tabulates the sorted
likelihood function L(x) in a way that itself uses Monte Carlo methods. The
technique uses a collection of n objects x, randomly sampled with respect to the
prior p, but also subject to an evolving constraint L(x) > L∗ preventing the
likelihood from exceeding the current limiting value L∗. 

We define xsi(x) = proportion of prior with likelihood greater than x.  In terms
of xsi, the objects are uniformly sampled subject to the constraint xsi < xsi∗,
where xsi∗ corresponds to L∗; 

      L  | +O++
         |      O++O
         |          +++O
      L* +...............+++
	 |               : ++
	 |               :   ++
	 |               :    ++
	 |               :     ++ 
	  ---------------+--------- xsi
	                xsi* 
Four objects (n=4) sampled uniformly in xsi < xsi∗, or equivalently in L > L∗.
(illustrated in the book in Fig.  9.3., p184)

At the outset, sampling is uniform over the entire prior, meaning that xsi∗=1
and L∗=0. The idea is then to iterate inwards in xsi and correspondingly upwards
in L, in order to locate and quantify the tiny region of high likelihood where
most of the joint distribution is to be found.


Adaptation into python of nested sampling from Sivia and Skilling 2006, p 188
Attempts to make its usage like emcee. (call it nestee? maybe not...)

Note that the code is very close to the original C code, which may not be
optimized for python usage.
"""


from math import *
import random
import numpy as np

DBL_MAX = 1e300

# ~U[0,1]
uniform = random.random

def plus(values):
	"""
	Logarithmic addition
	"""
	biggest = np.max(values)
	x = values - biggest
	result = np.log(np.sum(np.exp(x))) + biggest
	return result



def weighted_percentile(data, wt, percentiles): 
	"""Compute weighted percentiles. 
	If the weights are equal, this is the same as normal percentiles. 
	Elements of the C{data} and C{wt} arrays correspond to 
	each other and must have equal length (unless C{wt} is C{None}). 

	@param data: The data. 
	@type data: A L{np.ndarray} array or a C{list} of numbers. 
	@param wt: How important is a given piece of data. 
	@type wt: C{None} or a L{np.ndarray} array or a C{list} of numbers. 
		All the weights must be non-negative and the sum must be 
		greater than zero. 
	@param percentiles: what percentiles to use.  (Not really percentiles, 
		as the range is 0-1 rather than 0-100.) 
	@type percentiles: a C{list} of numbers between 0 and 1. 
	@rtype: [ C{float}, ... ] 
	@return: the weighted percentiles of the data. 
	""" 
	assert np.greater_equal(percentiles, 0.0).all(), "Percentiles less than zero" 
	assert np.less_equal(percentiles, 1.0).all(), "Percentiles greater than one" 
	data = np.asarray(data) 
	assert len(data.shape) == 1 
	if wt is None: 
		wt = np.ones(data.shape, np.float) 
	else: 
		wt = np.asarray(wt, np.float) 
		assert wt.shape == data.shape 
		assert np.greater_equal(wt, 0.0).all(), "Not all weights are non-negative." 
	assert len(wt.shape) == 1 
	n = data.shape[0] 
	assert n > 0 
	i = np.argsort(data) 
	sd = np.take(data, i, axis=0) 
	sw = np.take(wt, i, axis=0) 
	aw = np.add.accumulate(sw) 
	if not aw[-1] > 0: 
		raise ValueError, "Nonpositive weight sum" 
	w = (aw-0.5*sw)/aw[-1] 
	spots = np.searchsorted(w, percentiles) 
	o = [] 
	for (s, p) in zip(spots, percentiles): 
		if s == 0: 
			o.append(sd[0]) 
		elif s == n: 
			o.append(sd[n-1]) 
		else: 
			f1 = (w[s] - p)/(w[s] - w[s-1]) 
			f2 = (p - w[s-1])/(w[s] - w[s-1]) 
			assert f1>=0 and f2>=0 and f1<=1 and f2<=1 
			assert abs(f1+f2-1.0) < 1e-6 
			o.append(sd[s-1]*f1 + sd[s]*f2) 
	return o 

class Sample(object):
	def __init__(self):
		self.prior = None  # Uniform-prior controlling parameter for x
		self.pos   = None  # Geographical easterly position of lighthouse
		self.logL  = None  # logLikelihood = ln Prob(data | position)
		self.logWt = None  # log(Weight), adding to SUM(Wt) = Evidence Z

	def copy(self):
		ret = Sample()
		ret.__dict__ = self.__dict__.copy()
		return ret


class Model(object):
	def __init__(self, ndim, data):
		self.ndim = ndim
		self.pos  = np.empty(ndim, dtype=float)
		self.data = data

	def __len__(self):
		return self.ndim

	def lnp(self, pos):
		return 0.0	
	
	def fromPrior(self):
		"""
		Draw the parameters from the prior
		"""
		Obj = Sample()
	    	Obj.prior = uniform( len(self) ) # uniform in (0,1)
	    	Obj.logL  = self.lnp( Obj.pos )
		return Obj

	def proposal(self, guess, step):
		# Trial object
		Try = Sample() 
		Try.prior  = guess + step * ( 2. * uniform(len(self)) - 1. )  # |move| < step
		Try.prior -= np.floor(Try.prior)                        # wraparound to stay within (0,1)
		Try.pos    = Try.prior[:]
		return Try

	def explore(self, Obj, logLstar, m0 = 20, s0 = 0.1): 
		""" Evole object within likelihood constraints
		INPUTS:
			Obj		Sample	object to evole
			logLstar	float 	Likelihood constraint (L > Lstar)
		KEYWORDS:
			m0		int	pre-judged number of steps
			s0 		float	Initial guess suitable step-size in (0,1)
		OUTPUT:
			updated Obj

		Note unlike the original C version, this implementation returns an
		updated version of Obj rather than changing the original.
		"""
		ret = Obj.copy()
		step = s0      # Initial guess suitable step-size in (0,1)
		accept = 0     # # MCMC acceptances
		reject = 0     # # MCMC rejections
		Try = Sample() # Trial object

		for m in range(m0):  # pre-judged number of steps

		    # Trial object
		    Try = self.proposal(ret.prior, step)
		    Try.logL = self.lnp(Try.pos)  # trial likelihood value

		    # Accept if and only if within hard likelihood constraint
		    if Try.logL > logLstar:
			ret = Try.copy()
			accept += 1
		    else:
			reject += 1

		    # Refine step-size to let acceptance ratio converge around 50%
		    if( accept > reject ):   
			    step *= exp( 1.0 / accept )
		    if( accept < reject ):   
			    step /= exp( 1.0 / reject )
		return ret

class NestedSampler(object):
	def __init__(self, Mod):
		self.Model    = Mod

	def run_nested(self, Obj, max_iter):
		r = nested_sampling(max_iter, Obj, self.Model.explore)
		for k in r.keys():
			self.__dict__[k] = r[k]

	@property
	def lnprobability(self):
		return np.array( [(si.logL) for si in self.samples] ) # Proportional weight
		#return np.array( [exp(si.logWt - self.logZ) for si in self.samples] ) # Proportional weight

	@property
	def flatWeights(self):
		return np.array( [np.exp(si.logWt - self.logZ) for si in self.samples] ) # Proportional weight
		#return np.array( [exp(si.logWt - self.logZ) for si in self.samples] ) # Proportional weight

	@property
	def flatsamples(self):
		return np.array( [si.pos for si in self.samples] )

	def process_results(self):
		w    = self.flatWeights
		pos  = self.flatsamples
		mean = (pos * w[:,None]).sum(0) 
		sigval = [0.25,0.5,0.75]
		sig =  np.array([ weighted_percentile(pos[:,k], w, sigval ) for k in range(len(self.Model)) ])
		print( "# iterations: %i" % self.num_iterations )
		print( "Evidence: ln(Z) = %g +- %g" % (self.logZ, self.logZ_sdev) )
		print( "Information: H  = %g nats = %g bits" % ( self.info_nats, self.info_nats/log(2.0) ) )
		print( "mean position= %s" % ( mean )  )
		print( "Percentiles position:" )
		for k in range(len(sigval)):
			print( "p%d= %s" % ( sigval[k]*100., sig[:,k] ) )
	

# n = number of objects to evolve
def nested_sampling(max_iter, Obj, explore):
    """
    This is an implementation of John Skilling's Nested Sampling algorithm
    for computing the normalizing constant of a probability distribution
    (the posterior in Bayesian inference).

    The return value is a dictionary with the following entries:
        "samples"
        "num_iterations"
        "logZ"
        "logZ_sdev"
        "info_nats"
        "info_sdev"
    """

    Samples  = []       # Objects stored for posterior results
    logwidth = None     # ln(width in prior mass)
    logLstar = None     # ln(Likelihood constraint)
    H        = 0.0      # Information, initially 0
    logZ     = -DBL_MAX # ln(Evidence Z, initially 0)
    logZnew  = None     # Updated logZ
    copy     = None     # Duplicated object
    worst    = None     # Worst object
    nest     = None     # Nested sampling iteration count
    n        = len(Obj)

    # Outermost interval of prior mass
    logwidth = log(1.0 - exp(-1.0 / n))

    # NESTED SAMPLING LOOP 
    for nest in range(max_iter):

        # Worst object in collection, with Weight = width * Likelihood
        worst = 0
        for i in range(1,n):
            if Obj[i].logL < Obj[worst].logL:
                worst = i

        Obj[worst].logWt = logwidth + Obj[worst].logL

        # Update Evidence Z and Information H
        logZnew = plus([logZ, Obj[worst].logWt])
        H = exp(Obj[worst].logWt - logZnew) * Obj[worst].logL + \
            exp(logZ - logZnew) * (H + logZ) - logZnew
        logZ = logZnew

        # Posterior Samples (optional)
        Samples.append(Obj[worst])

        # Kill worst object in favour of copy of different survivor
        if n>1: # don't kill if n is only 1
            while True:
                copy = int(n * uniform()) % n  # force 0 <= copy < n
                if copy != worst:
                    break

        logLstar = Obj[worst].logL;       # new likelihood constraint
        Obj[worst] = Obj[copy];           # overwrite worst object

        # Evolve copied object within constraint
        updated = explore(Obj[worst], logLstar)
        assert(not updated is None) # Make sure explore didn't update in-place
        Obj[worst] = updated

        # Shrink interval
        logwidth -= 1.0 / n

    # Exit with evidence Z, information H, and optional posterior Samples
    sdev_H = H/log(2.)
    sdev_logZ = sqrt(H/n)
    return {"samples":Samples, 
            "num_iterations":(nest+1), 
            "logZ":logZ,
            "logZ_sdev":sdev_logZ,
            "info_nats":H,
            "info_sdev":sdev_H}

