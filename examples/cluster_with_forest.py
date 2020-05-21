from gpu_rpfk.RPFK import RPFK

import numpy as np
import matplotlib.pyplot as plt
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters used to init experiments")
    parser.add_argument("-n", "--n_points", help="Number of points",
                        type=int, default=2**13)
    parser.add_argument("-d", "--n_dim", help="Number of points",
                        type=int, default=2)
    parser.add_argument("-v", "--rpfk_verbose", help="RPFK verbose level",
                        type=int, default=1)
    


    args = parser.parse_args()

    n_points = args.n_points
    n_dim = args.n_dim
    rpfk_verbose = args.rpfk_verbose

    points = np.random.random((n_points,n_dim))

    min_tree_children = int(n_points/30)
    max_tree_children = 3*min_tree_children

    print("Number of points {}".format(n_points))
    print("Number of dimensions {}".format(n_dim))

    rpfk = RPFK(random_state=0)

    result = rpfk.cluster_by_sample_tree(points,
                                         min_tree_children=min_tree_children,
                                         max_tree_children=max_tree_children,
                                         max_tree_depth=5000,
                                         random_motion_force=0.1,
                                         verbose=rpfk_verbose)

    total_leaves, max_child, nodes_buckets, bucket_sizes = result
        
    print("Total leaves: {}".format(total_leaves))

    offset = 0
    fig, ax = plt.subplots(1, 1, sharey=True, tight_layout=True)
    while offset < len(nodes_buckets):
        bucket_slice = nodes_buckets[offset:offset+max_child]
        # -1 values represent empty slots on buckets
        bucket_slice = bucket_slice[np.where(bucket_slice) != -1]
        ax.scatter(points[bucket_slice,0], points[bucket_slice,1], s=5, alpha=0.8)
        offset+=max_child
    
    print("Min/Max bucket size: {}/{}".format(min(bucket_sizes), max(bucket_sizes)))
    
    plt.show()