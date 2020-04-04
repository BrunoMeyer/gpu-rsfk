import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RPFK execution into a histogram with leaves count size")
    parser.add_argument("-f", "--file", type=str, help="Log file path", required=True)
    parser.add_argument("-b", "--total_bins", type=str, help="Number of bins in histogram",
                        required=False, default="auto")
    
    args = parser.parse_args()

    file_name = args.file
    total_bins = args.total_bins

    trees = []
    with open(file_name, "+r") as f:
        new_tree = []
        for line in f.readlines():
            if len(new_tree) > 0 and not "\t" in line:
                trees.append(new_tree)
                new_tree = []
            if "\t" in line:
                leaf_size = int(line.split("\t")[0])
                leaf_count = int(line.split("\t")[1])
                new_tree.append([leaf_size, leaf_count])

    # Sum of histograms
    trees.append(np.concatenate(tuple([t for t in trees])))
    trees = np.array(trees)

    fig, axis = plt.subplots(int(np.ceil((len(trees))/2)),2,
                             figsize=(16,16))
    axis = np.array(axis).reshape((-1,1))[:,0]
    for i,(ax,tree) in enumerate(zip(axis, trees)):
        tree = np.array(tree)
        if total_bins == "auto":
            tmp_bins = len((set(tree[:,0])))
        else:
            tmp_bins = int(total_bins)
        # the histogram of the data
        n, bins, patches = ax.hist(tree[:,0],
                                   tmp_bins,
                                   weights=tree[:,1],
                                   density=False,
                                   facecolor='g',
                                   alpha=0.75)


        ax.set_xlabel('Leaf size')
        ax.set_ylabel('Leaf count')
        if i != len(trees)-1:
            ax.set_title('Tree number {}'.format(i))
        else:
            ax.set_title('Accumulated histogram')
    
    plt.subplots_adjust(hspace=0.5)
    plt.show()
