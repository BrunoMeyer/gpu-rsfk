# GPU Random Sample Forest KNN (RSFK)

This project presents a CUDA implementation of the Random Sample Forest KNN (RSFK) algorithm.

# Citation

Please cite the corresponding papers if it was useful for your research:

```bibtex
@article{meyer2022global,
  title={Global and local structure preserving GPU t-SNE methods for large-scale applications},
  author={Meyer, Bruno Henrique and Pozo, Aurora Trinidad Ramirez and Zola, Wagner M Nunan},
  journal={Expert Systems with Applications},
  volume={201},
  pages={116918},
  year={2022},
  publisher={Elsevier}
}

@article{meyer2021improving,
  title={Improving barnes-hut t-sne algorithm in modern gpu architectures with random forest knn and simulated wide-warp},
  author={Meyer, Bruno Henrique and Pozo, Aurora Trinidad Ramirez and Zola, Wagner M Nunan},
  journal={ACM Journal on Emerging Technologies in Computing Systems (JETC)},
  volume={17},
  number={4},
  pages={1--26},
  year={2021},
  publisher={ACM New York, NY}
}

```bibtex
@inproceedings{meyer2021warp,
  title={Warp-centric k-nearest neighbor graphs construction on GPU},
  author={Meyer, Bruno and Pozo, Aurora and Nunan Zola, Wagner M},
  booktitle={50th International Conference on Parallel Processing Workshop},
  pages={1--10},
  year={2021}
}
```

## K-NNG
The K-Nearest Neighbor Graph is a problem that consists in: For each element of a set V, find the K most similar objects (neighbors) to each object that also is contained in V.
The NN-Descent algorithm [[1]](#references) is an algorithm that computes an approximation of K-NNG using MapReduce primitives.

In this project, we provide an implementation of RSFK algorithm in CUDA language with a python interface. RSFK uses a very similar strategy to [LarveVis](https://github.com/lferry007/LargeVis) [[3]](#references) algorithm, where randomized trees generate different partitions. These partitions are used to explore real neighbors for every point. Also, a neighborhood exploration method is used to improve the quality of the K-NNG approximation.


## Compiling and Installing
```bash
make && sudo make install # Generic compilation and global install
make && make install SETUP_FLAG="--user" # Generic compilation and local install
make python_arch75 && make install SETUP_FLAG="--user" # Compilation for a specific NVIDIA architecture and local install
```

## Run (Python)
### Constructions K-Nearest Neighbor Graph
```python
import numpy as np
from gpu_rsfk.RSFK import RSFK
points = np.random.random((1000000,100))
K = 32 # number of neighbors
rsfk = RSFK(random_state=0)
indices, dist = rsfk.find_nearest_neighbors(points,
                                            K,
                                            verbose=1,
                                            n_trees=50) # number of trees
print(indices) # the neighborhood of each point
print(dist) # the squared distance to each neighbor
```

```python
[[     0  31675 681056 ...  76565 215445 466714]
 [     1 788655 291762 ... 518101 372516 261391]
 [     2 569866 858381 ... 960535 794887 159680]
 ...
 [999997 625514 233354 ... 217298 568597 805424]
 [999998 588030 796374 ... 611888 321668 748140]
 [999999  63518 387808 ...  18947 277399 995385]]
[[ 0.         9.997955  10.742531  ... 11.311033  10.854658  10.957819 ]
 [ 0.        11.359983  11.517796  ... 10.417099  11.3883095 11.174629 ]
 [ 0.        11.319759  11.306412  ... 11.440113  11.263405   9.516406 ]
 ...
 [ 0.        11.373617   9.993327  ... 10.597631  11.374975  10.346858 ]
 [ 0.        10.520063  10.235972  ...  9.641689  10.381821   9.694969 ]
 [ 0.        11.417332  11.84985   ... 11.933416  10.799774  11.2241   ]]
```

### Construction a partition with a Random Sample Tree
The following code contains an example of how to build a partition of a set of points from a Random Sample Tree.
Also, the file [``examples/cluster_with_forest.py``](https://github.com/BrunoMeyer/gpu-rsfk/blob/master/examples/cluster_with_forest.py) contains the code to create [figures from the result](#random-sample-forest-knn).

```python
import numpy as np
from gpu_rsfk.RSFK import RSFK
points = np.random.random((10000,2))
rsfk = RSFK(random_state=0)
result = rsfk.cluster_by_sample_tree(points,
                                     min_tree_children=256,
                                     max_tree_children=1024,
                                     verbose=1)
total_leaves, max_child, nodes_buckets, bucket_sizes = result
print(total_leaves) # total of buckets
print(max_child) # maximum size of a bucket
print(nodes_buckets) # the partition of points serialized in one vector
print(bucket_sizes) # the size of each subset (bucket) on the partition
```

```python
Creating cluster with forest takes 0.109001 seconds
16
819
[1255 1262 1265 ...   -1   -1   -1]
[819 469 657 269 642 420 435 762 749 780 665 520 816 425 806 766]

```


## Random Sample Forest KNN
![Partition of points generated by a Random Sample Tree](docs/img/cluster_tree.png)

The RSFK algorithm presented in this project consists in combining the result of different "weak" approximations like the Random Projection Forest KNN algorithm [[2]](#references). Also, RSFK uses a very similar strategy like that described in [LarveVis](https://github.com/lferry007/LargeVis) paper [[3]](#references), where the [ANNOY](https://github.com/spotify/annoy) project is used to construct the K-NNG.

The Random Projection Forest KNN creates different trees. Each tree creates a partition of the set of points with *D* dimensions, dividing the points in different subsets. Each point considers all other points that are in its subset as potential neighbors. This limitation of exploration reduces the quadratic computational time complexity of the exact algorithm, leading to an approximation that is sufficient when several trees are created. Each tree is created as follow:


- Create a random direction in the D-dimensional space;

- Project each point of the set in the random direction;

- Split the set into two new subsets considering the median projection values;

- Repeat the process for each subset until the subset size reaches a threshold.
 
The main difference between Random Projection Forests and RSFK lies in the fact that the trees do not project the points into a random projection. Instead, RSFK only samples two random points, generates a hyperplane equally distant to those points, and creates the two subsets based in the side that each point is in the hyperplane. This process allows an implementation that demands less computational overhead, reducing the computational time and the simplification of the parallelization.

### Nearest Neighbor Exploration
Like [LarveVis](https://github.com/lferry007/LargeVis) [[3]](#references), we also present a post-processing technique that explore the neighbors of neighbors.
This exploration can be executed many times and is controlled by the parameter ``nn_exploring_factor``.

## Similarity Search

Another option to create the K-NNG from a set of points is to use algorithms that compute the generic similarity search considering a set base of points and a set of query points.
An approximation can be created using approximations of KNN like [FAISS library](https://github.com/facebookresearch/faiss).


The following code is an example of the usage of FAISS with GPU.

```python
import faiss
import numpy as np

points = np.random.random((1000000,100))
K = 32

res = faiss.StandardGpuResources()  # use a single GPU
# Set the base and query points as the same 
xb = np.require(points, np.float32, ['CONTIGUOUS', 'ALIGNED'])
xq = np.require(points, np.float32, ['CONTIGUOUS', 'ALIGNED'])
nq, d = xq.shape

# Parameters that control the trade-off between Quality and Time  
nlist = int(np.sqrt(1000000))
nprobe = 20

quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
index.nprobe = nprobe

index.train(xb)
index.add(xb)
index = faiss.index_cpu_to_gpu(res, 0, index)

indices, dist = index.search(xq, K)     # actual search

print(indices)
print(dist)
```

# References

[1] Dong, W., Moses, C. and Li, K., 2011, March. Efficient k-nearest neighbor graph construction for generic similarity measures. In Proceedings of the 20th international conference on World wide web (pp. 577-586).

[2] Yan, D., Wang, Y., Wang, J., Wang, H. and Li, Z., 2019. K-nearest Neighbors Search by Random Projection Forests. IEEE Transactions on Big Data.

[3] Tang, J., Liu, J., Zhang, M. and Mei, Q., 2016, April. Visualizing large-scale and high-dimensional data. In Proceedings of the 25th international conference on world wide web (pp. 287-297).
