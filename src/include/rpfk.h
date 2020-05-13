#include "common.h"

// Class used to construct Random Projection forest and execute KNN
class RPFK
{
public:
    
    // Data points. It will be stored as POINTS x DIMENSION or
    // DIMENSION x POINTS considering the defined POINTS_STRUCTURE
    typepoints* points; 
    
    // Indices of the estimated k-nearest neighbors for each point
    // and squared distances between
    // It can be previously initialized before the execution of RPFK
    // with valid indices and distances, otherwise it must assume that indices
    // have -1 value and distances FLT_MAX (or DBL_MAX)
    // The indices ARE NOT sorted by the relative distances
    int* knn_indices;
    typepoints* knn_sqr_distances;
    
    // Maximum and Minimum number of points that will be present in each leaf node (bucket)
    // This affect the local KNN step after the construction of each tree
    int MIN_TREE_CHILD;
    int MAX_TREE_CHILD;

    // The limit of the tree depth. The algorithm may break if a tree reach
    // a higher value than this parameter. When the tree is 'ready',
    // there is a early stop and this limit will not be reached
    // Known Bug: It may be necessary to add a 'random motion' before the
    // execution of the algorithm to prevent that two points be positioned
    // in the the same location, leading to the impossibility of the
    // tree construction and the reach of MAX_DEPTH 
    int MAX_DEPTH;

    // SEED used to the pseudorandom number generator
    int RANDOM_SEED;

    // Nearest neighbor exploring is a pos-processing technique that will improve
    // the accuracy of the approximated KNN. The neighbors of the neighbors
    // will be treated as neighbor candidates for each point, which is a
    // O(N*D*(K^2)) time cost step. This can be executed several times using the
    // updated indices as the new input, and this ammount is represented by
    // nn_exploring_factor. When nn_exploring_factor=0 the Nearest Neighbor Exploring
    // will not be executed
    int nn_exploring_factor;
    
    RPFK(typepoints* points,
         int* knn_indices,
         typepoints* knn_sqr_distances,
         int MIN_TREE_CHILD,
         int MAX_TREE_CHILD,
         int MAX_DEPTH,
         int RANDOM_SEED,
         int nn_exploring_factor):
         points(points),
         knn_indices(knn_indices),
         knn_sqr_distances(knn_sqr_distances),
         MIN_TREE_CHILD(MIN_TREE_CHILD),
         MAX_TREE_CHILD(MAX_TREE_CHILD),
         MAX_DEPTH(MAX_DEPTH),
         RANDOM_SEED(RANDOM_SEED),
         nn_exploring_factor(nn_exploring_factor){}
    

    // Create a random projection tree and update the indices by considering
    // the points in the same leaf node as candidates to neighbor
    // This ensure that each point will have K valid neighbors indices, which
    // can be very inaccurate.
    // The device_knn_indices parameter can be previously initialized with
    // valid indices or -1 values. If it has valid indices, also will be necessary
    // to add the precomputed squared distances (device_knn_sqr_distances) 
    void add_random_projection_tree(thrust::device_vector<typepoints> &device_points,
                                    thrust::device_vector<int> &device_knn_indices,
                                    thrust::device_vector<typepoints> &device_knn_sqr_distances,
                                    int K, int N, int D, int VERBOSE,
                                    string run_name);
    
    
    // Run n_tree times the add_random_projection_tree procedure and the nearest
    // neighbors exploring if necessary
    void knn_gpu_rpfk_forest(int n_trees,
                             int K, int N, int D, int VERBOSE,
                             string run_name);
};