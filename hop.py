A = """
.XXX.
X...X
XXXXX
X...X
X...X
"""

Z = """
XXXXX
...X.
..X..
.X...
XXXXX
"""
import numpy as np
from numpy import array
from pylab import imshow, cm, show
from numpy import zeros, outer, diag_indices 
from numpy import vectorize, dot

def to_pattern(letter):
	 return array([+1 if c=='X' else -1 for c in letter.replace('\n','')])

def display(pattern):
    imshow(pattern.reshape((5,5)),cmap=cm.binary, interpolation='nearest')
    show()

patterns = array([to_pattern(A), to_pattern(Z)])

def train(patterns):
    r,c = patterns.shape
    W = zeros((c,c))
    for p in patterns:
        W = W + outer(p,p)
    W[diag_indices(c)] = 0
    return W/r

def recall(W, patterns, steps=5):
    sgn = vectorize(lambda x: -1 if x<0 else +1)
    for _ in xrange(steps):        
        patterns = sgn(dot(patterns,W))
	display(patterns)
    return patterns

def hopfield_energy(W, patterns):
    from numpy import array, dot
    return array([-0.5*dot(dot(p.T,W),p) for p in patterns])

pattern1 = array([-1 if c==1 else 1 for c in to_pattern(A)[0:8]])
pattern1 = np.append(pattern1, to_pattern(A)[8:25])
display(pattern1)
display(recall(train(patterns), pattern1))



