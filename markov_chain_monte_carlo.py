import numpy as np
from numpy.random import multivariate_normal
from numpy.random import uniform
from functools import partial
import scipy.integrate as integrate

def markov_seq(u, start, g):
    """ markov_seq yield a Markov sequence,
        input:   u = probability density 
                 start = value where the chain start
                 g = generator of a symmetric transition distribution
                 """                   
    x = start            
                 
    while True:
            
        y = x + g() 
            
        mu_x = max(u(x), 0.0)
        mu_y = max(u(y), 0.0)
        
        if uniform(0.0, mu_x) < mu_y:  
            x = y
        yield x             
            
def estim_mean_k(markov_chain, k, a, b):   
    """ estim_mean_k returns for MCMC an estimation of the 
        mean based on the kth batch
        input: markov_chain = Markov Chain
               k = kth batch
               a = number of batches
               b = size of a batch      """
    mean_k = 0
    for i in range((k-1)*b,(k*b)):         #we don't begin at (k-1)*b+1 because markov_chain begins at 0
        mean_k += markov_chain[i]
    return mean_k / b

def metropolis_hastings(f, u, dim, start = None, g = None, cov = None):
    """ metropolis_hastings return a tuple of approximation of the integrale and the approximaiton of the error,
        input:   f = integrable function
                 u = probability density 
                 dim = dimension
                 start = starting point of Markov Chain
                 g = generator of a symmetric transition distribution
                 cov = covariance matrix of the distribution g"""      
    
    if start is None:
        start = np.array([0.0] * dim) 
        
# if g is None we use a mutivariate normal law
    if g is None:
        centre = [0.0] * dim
        if cov is None:
            cov = 1.0 * np.identity(dim)
        g = partial(multivariate_normal, centre, cov)
        
# Initialisation of the generator
    generator = markov_seq(u,start,g)

# *prime* the Markov Chain
    for _ in range(0,10000):   
        next(generator)   

#creation of the Markov Chain
    N = 10000                  # N = a * b
    a = 100
    b = 100
    markov_result = list()
    f_markov_result = list()       
    
    for i in range(0,N):
        value = next(generator)
        markov_result.append(value)  
        f_markov_result.append(f(value))
        
# batch means of f_markov_result         
    E_list_mean = list()   
    for k in range(0,a):     
        E_list_mean.append(estim_mean_k(f_markov_result, k, a, b))   
               
# approximation of the integrale 
    integrent = integrate.quad(u,- np.inf, np.inf)
    if integrent[0] != 1 and integrent[1] > 1:      #verifying if u is a probability density
        E = integrent[0] * (sum(E_list_mean)/len(E_list_mean))
    else:
        E = sum(E_list_mean)/len(E_list_mean)     
        
# batch means of markov_result
    list_mean = list()   
    for k in range(0,a):     
        list_mean.append(estim_mean_k(markov_result, k, a, b))  
         
#estimation of the variance 
    var = 0
    for i in range(a):
        var += np.square(( list_mean[i] - E ))   
    var = var * ( b / (a-1) )       
    
#batch means estimate of the Monte Carlo standard error is
    error = np.sqrt( var / N )

# We return the tuple of result, Monte Carlo standard error          
    return (E, error[0])        
    
    
    



#======================================================  
#      integrable function  
#====================================================== 
def maximum(x):
    return np.maximum(x)
    
def norm(x) :
    return np.sum(x**2)

def in_ball(x) :
    return norm(x) < 1.

def just_sum(x) :
    return np.sum(x)

def poly(x) :
    if isinstance(x, (int, float)) :
        return x**4
    else :
        pwrs = np.arange(len(x))
    return np.sum(x ** pwrs)
  
    
    
#======================================================  
#     probability density
#======================================================   
   
    
def uniforme_pdf(a,b,x):
    """ The probability density function of the uniform
        distribution on the interval [a,b]^dim.
        input : x = 1-dim numpy array """
    if isinstance(x, (int, float)) :
        prob = 1./(b-a)
        if x < a or b < x :
            return 0.
    else :
        dim = len(x)
        prob = 1./(b-a)**dim
        for i in range(len(x)) :
            if x[i] < a or b < x[i] :
                return 0.
    return prob

dim = 4 
a = 0
b = 10    
f = norm   
u = partial( uniforme_pdf, a, b) 
   
   
#print(metropolis_hastings(f, u, dim, start = None, g = None, cov = None))



def norm_pdf(centre, x) :
    """ The probability density function of the standard
    normal with covariance = 1 and given centere.""" 
    if isinstance(x, (int, float)) :
        return np.exp(-(x-centre)**2/2)/np.sqrt(2*np.pi)
    else :
        return np.exp(-norm(x-centre)/2)/np.sqrt(2*np.pi)

dim = 4 
centre = np.array([0.0] * dim) 
f = norm       
u = partial(norm_pdf, centre)

#print(metropolis_hastings(f, u, dim, start = None, g = None, cov = None))




#======================================================  
#      symmetric transition distribution
#======================================================  

a = 2.
b = 3.
dim = 4
f = norm   
u = partial( uniforme_pdf, a, b) 
g = partial(np.random.uniform, a, b, dim)

#print(metropolis_hastings(f, u, dim, start = None, g = g, cov = None))


def random_from_ball(dim, rad) :
    p = 1 + rad**2
    while norm(p) > rad**2 :    
        p = np.random.uniform(-rad, rad, dim)
    if dim == 1 :
        return p[0]
    return p

rad = 0.3
f = norm   
u = partial( uniforme_pdf, a, b) 
g = partial(random_from_ball, dim, rad)

#print(metropolis_hastings(f, u, dim, start = None, g = g, cov = None))



#======================================================  
#      Test: metropolis_hastings  
#======================================================     


rad = 0.3
dim = 1
centre = np.array([0.0] * dim)
f = just_sum   
u = partial(norm_pdf, centre) 
g = partial(random_from_ball, dim, rad)
start = np.array([2.2] * dim) 
 
def test_metropolis_hastings():
    assert [metropolis_hastings(f, u, dim, start = None, g = g, cov = None) < (0.,1.)
             or metropolis_hastings(f, u, dim, start, g = g, cov = None) > (0.,1.) ]
    print("Passed metropolis_hastings value check")
    
    
def test__metropolis_hastings_dimension(dimension):
    for i in range(1,dimension+1) : 
        print(metropolis_hastings(f, u, dimension, start = None, g = None, cov = None))
        
        
test_metropolis_hastings()
dimension = 3
test__metropolis_hastings_dimension(dimension)