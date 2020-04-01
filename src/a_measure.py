import scipy.stats as ss
import numpy as np
# equation to calculate A is differnet to original but formally equivalent https://mtorchiano.wordpress.com/2014/05/19/effect-size-of-r-precision/
# prevents imprecision caused by order of operations
# basically this puts division at the end after the +-* have all been completed to avoid rounding errors

'''
Arguments
a, b - numpy arrays

returns
Vargha Delaney A Statistic for Effect Size between the two arrays
'''
def a_measure(a, b):
    m = len(a)
    n = len(b)

    c = np.concatenate((a, b))
    r = ss.rankdata(c)
    r1 = sum(r[0:m])

    A = (2*r1 - m*(m+1)) / (2*m*n)

    return A
