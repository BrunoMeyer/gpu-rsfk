import numpy as np
from gpu_rsfk.RSFK import RSFK
# points = np.random.random((1000000,100))
points = np.random.random((10000,100))
K = 32 # number of neighbors
rsfk = RSFK(random_state=0)
indices, dist = rsfk.find_nearest_neighbors(points,
                                            K,
                                            verbose=1,
                                            n_trees=50) # number of trees
print(indices) # the neighborhood of each point
print(dist) # the squared distance to each neighbor
