from numpy import save, array
from numpy.random import randn
save('matrix', array(randn(2 ** 15, 2 ** 15), 'float16'))
