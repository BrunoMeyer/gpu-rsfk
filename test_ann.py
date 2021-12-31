import numpy as np
from gpu_rsfk.RSFK import RSFK

np.random.seed(0)

NB = 10000000
NQ = 1000000
D = 32

points = np.random.random((NQ,D))
query_points = np.random.random((NQ,D))

# points = np.arange(NB*D).reshape(-1, D)
# query_points = np.arange(NQ*D).reshape(-1, D)

K = 8 # number of neighbors
rsfk = RSFK(random_state=0)

indices, dist = rsfk.find_nearest_neighbors_ann(
    points,
    query_points,
    K,
    verbose=2,
    n_trees=2,
    min_tree_children=K+1,
    add_bit_random_motion=True,
    max_tree_children=3*(K+1)) # number of trees

print(indices) # the neighborhood of each point
print(dist) # the squared distance to each neighbor
