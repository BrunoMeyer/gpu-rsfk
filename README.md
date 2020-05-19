# GPU Random Sample Forest KNN (RSFK)

This project presents a CUDA implementation of the Random Sample Forest KNN (RSFK) algorithm.

## K-NNG
The K-Nearest Neighbor Graph is a problem that consists in: For each element of a set V, find the K most similar objects (neighbors) to each object that also is contained in V.
The NN-Descent algorithm [[1]](#references) is an algorithm that computes an approximation of K-NNG using MapReduce primitives.

## Random Sample Forest KNN
The RSFK algorithm presented in this project consists in combining the result of different "weak" approximations like the Random Projection Forest KNN algorithm [[2]](#references). Also, RSFK uses a very similar strategy like that described in [LarveVis](https://github.com/lferry007/LargeVis) paper, where the [ANNOY](https://github.com/spotify/annoy) project is used to construct the K-NNG.

The Random Projection Forest KNN creates different trees. Each tree creates a partition of the set of points with *D* dimensions, dividing the points in different subsets. Each point considers all other points that are in its subset as potential neighbors. This limitation of exploration reduces the quadratic computational time complexity of the exact algorithm, leading to an approximation that is sufficient when several trees are created. Each tree is created as follow:


- Create a random direction in the D-dimensional space;

- Project each point of the set in the random direction;

- Split the set into two new subsets considering the median projection values;

- Repeat the process for each subset until the subset size reaches a threshold.
 
The main difference between Random Projection Forests and RSFK lies in the fact that the trees do not project the points into a random projection. Instead, RSFK only samples two random points, generates a hyperplane equally distant to those points, and creates the two subsets based in the side that each point is in the hyperplane. This process allows an implementation that demands less computational overhead, reducing the computational time and the simplification of the parallelization.


## Compiling and Installing
```bash
make && sudo make install
make && make install SETUP_FLAG="--user"
make python_arch75 && make install SETUP_FLAG="--user"
make main && ./rpfk
make main_arch75 && ./rpfk
```


## Similarity Search


# References

[1] Dong, W., Moses, C. and Li, K., 2011, March. Efficient k-nearest neighbor graph construction for generic similarity measures. In Proceedings of the 20th international conference on World wide web (pp. 577-586).

[2] Yan, D., Wang, Y., Wang, J., Wang, H. and Li, Z., 2019. K-nearest Neighbors Search by Random Projection Forests. IEEE Transactions on Big Data.

[3] Tang, J., Liu, J., Zhang, M. and Mei, Q., 2016, April. Visualizing large-scale and high-dimensional data. In Proceedings of the 25th international conference on world wide web (pp. 287-297).
