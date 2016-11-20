# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from numpy.random import multivariate_normal
#from numpy.random import normal
from numpy.random import random
from numpy import zeros
from numpy import eye   
from random import randint
#from scipy import stats


def binary_decimal(m,n) :
    """ Given integers 0 <= m <= n, n != 0, yields
    the (shortest) binary decimal expansion of m/n """
    assert ( isinstance(m, int) and isinstance(n, int) and
        0 <= m <= n and n != 0 )
    rest = m
    while rest != 0 :
        rest *= 2
        coeff = rest // n
        yield coeff
        rest -= coeff * n


def biased_coin(m,n) :
    """ Using random.randint(0, 1) as a fair coin, this
    returns 1 with probablity m/n, where m,n must be
    non-negative integers with m <= n. """
    # binary_decimal checks m,n for us
    bin_dec = binary_decimal(m,n)
    for x in bin_dec :
        if x != randint(0, 1) :
            return x
    return 0 # if binary_decimal is exhausted





def markov_seq(u, n, g = None, A = None):
    """ markov_seq yield a Markov sequence,
        input:   u = probability density 
                 g = generator of a symmetric transition distribution 
                 n = dimension
                 A = compact support of u"""
    
    x = random( size = n )  # the first element of the sequence chosen randomly 
    seq = [x]               # Markov sequence of 1 element
    while True:    
        g = multivariate_normal( zeros(n), eye(n))   # an element of the multi-normal-dist
        y = x + g
        minimum = min(1,(u(y)/u(x)))
        if minimum == 1:
            coin = biased_coin(1,1)
        elif minimum == u(y)/u(x):
            coin = biased_coin(u(y),u(x))
        if coin == 1:
            x_next = y
        elif coin != 1:
            x_next = x
        seq.append(x_next)
        yield seq[1]



def uniforme(x):
    for i in range(len(x)):
        if ( x[i] < -5 ) and ( x[i] > 5 ):
            return 0
    return 1/10
    


print(next(markov_seq(uniforme,3)))       













