Nested Sampling
===============

This is my playground around *Nested Sampling methods*

I adapted the C-code from Skilling, 2006 into python. 
I adapted the C-code of nested sampling from Sivia and Skilling 2006, p 188
Attempts to make its usage like emcee. (call it nestee? maybe not...)

*Note that the code is very close to the original C code, which may not be
optimized for python usage. Although not python-optimized this works just fine.
*


Background
----------

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

Example for many objects sampled uniformly in x < x1, or equivalently
in L > L∗.
![Nest](http://www.inference.phy.cam.ac.uk/bayesys/box/figs/nest.gif)
(Illustration from Skilling, 2006)

At the outset, sampling is uniform over the entire prior, meaning that xsi∗=1
and L∗=0. The idea is then to iterate inwards in xsi and correspondingly upwards
in L, in order to locate and quantify the tiny region of high likelihood where
most of the joint distribution is to be found.




**nest.py** is the main code (adapation from Sivia and Skilling 2006)
**lighthouse.py**  is an application of nested sampling to the Light House 
                        problem from Sivia and Skilling 2006, p 192

