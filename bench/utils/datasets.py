import pickle
import numpy as np
from sklearn.datasets import load_digits, load_iris
import os
from sklearn.neighbors import NearestNeighbors
import faiss

import time

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

KNN_CACHE_DIR = "./.cache/knn/"
ARTIFICIAL_DATASET_DIR_CACHE = "./.cache/artificial_datasets/"


def unpickle(file):
    '''Load byte data from file'''
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='latin-1')
        return data

def load_dataset(
    name,
    random_seed=0,
    npoints=0,
    ndim=0,
    ):
    if name == 'MNIST':
        mnist = fetch_openml('mnist_784', version=1)
        X = mnist.data
        y = mnist.target
        logger.info(f"Loaded MNIST dataset with {X.shape[0]} samples and {X.shape[1]} features.")
    elif name == 'ARTIFICIAL_UNIFORM':
        cache_path = os.path.join(
            ARTIFICIAL_DATASET_DIR_CACHE,
            f"{name}_{npoints}_{ndim}_{random_seed}.pickle"
        )
        os.makedirs(ARTIFICIAL_DATASET_DIR_CACHE, exist_ok=True)

        if os.path.isfile(cache_path):
            with open(cache_path, "rb") as fo:
                X = pickle.load(fo)
            logger.info(f"Loaded artificial uniform dataset from cache with {X.shape[0]} samples and {X.shape[1]} features.")
            return X, None
        n_samples = npoints
        n_features = ndim
        # generate uniform [0,1) float32 dataset
        # X = np.random.default_rng(random_seed).random((n_samples, n_features), dtype=np.float32)
        X = np.random.rand(n_samples, n_features).astype('float32')
        y = None
        logger.info(f"Created artificial uniform dataset with {X.shape[0]} samples and {X.shape[1]} features.")
        with open(cache_path, "wb") as fo:
            pickle.dump(X, fo)
    else:
        logger.warning(f"Dataset {name} is not supported.")

    return X, y

def load_dataset_knn(
    name,
    max_k=128,
    npoints=0,
    ndim=0,
    random_seed=0,
    ):
    
    # knn_file = DATASETHOME+"/"+name+"_knn.pickle"
    knn_file = os.path.join(
        KNN_CACHE_DIR,
        f"{name}_knn_{npoints}_{ndim}_{random_seed}.pickle"
    )
    os.makedirs(KNN_CACHE_DIR, exist_ok=True)
    
    if os.path.isfile(knn_file):
        data = unpickle(knn_file)
        if data[0].shape[1] >= max_k:
            return data[0][:,:max_k], data[1][:,:max_k]

    logger.info(f"Computing exact knn k={max_k} with faiss for dataset {name}...")
    dataX, dataY = load_dataset(name, random_seed=random_seed, npoints=npoints, ndim=ndim)

    logger.info(f"Dataset has {dataX.shape[0]} samples and {dataX.shape[1]} features.")


    res = faiss.StandardGpuResources()  # use a single GPU
    res.setTempMemory(256 * 1024 * 1024)  # 256MB, for example

    # xb = np.require(dataX, np.float32, ['CONTIGUOUS', 'ALIGNED'])
    # xq = np.require(dataX, np.float32, ['CONTIGUOUS', 'ALIGNED'])

    # nq, d = xq.shape
    nq, d = dataX.shape

    # nlist = int(np.sqrt(nq))
    # logger.info(f"Using nlist={nlist} for faiss index.")

    init_t = time.time()
    index = faiss.IndexFlatL2(d)
    index = faiss.index_cpu_to_gpu(res, 0, index)

    # here we specify METRIC_L2, by default it performs inner-product search
    logger.info("Training faiss index...")
    index.train(dataX)
    logger.info("Adding vectors to faiss index...")
    index.add(dataX)                  # add may be a bit slower as well
    # distances, indices = index.search(xq, max_k)     # actual search
    distances, indices = index.search(dataX, max_k)     # actual search
    logger.info(f"Faiss knn search completed in {time.time()-init_t:.2f} seconds.")


    fo = open(knn_file, "wb")
    pickle.dump((distances,indices),fo)
    return distances, indices
