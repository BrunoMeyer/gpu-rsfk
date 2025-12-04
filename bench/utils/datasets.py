import pickle
import numpy as np
from sklearn.datasets import load_digits, load_iris
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import fetch_openml
import faiss
import openml
import torch
from torchvision import transforms as T, models, datasets as tv_datasets
from torch.utils.data import DataLoader, Subset
from sklearn.decomposition import TruncatedSVD

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
    dataset_id=None,
    random_seed=0,
    npoints=0,
    ndim=0,
    ):
    if name == 'MNIST':
        cache_path = os.path.join(
            f"./.cache/mnist.pickle"
        )
        if os.path.isfile(cache_path):
            with open(cache_path, "rb") as fo:
                X = pickle.load(fo)
            logger.info(f"Loaded MNIST dataset from cache with {X.shape[0]} samples and {X.shape[1]} features.")
            return X, None
        mnist = fetch_openml('mnist_784', version=1)
        X = mnist.data
        y = mnist.target
        logger.info(f"Loaded MNIST dataset with {X.shape[0]} samples and {X.shape[1]} features.")
    elif name == 'KDDCUP99':
        cache_path = os.path.join(
            f"./.cache/kddcup99.pickle"
        )
        if os.path.isfile(cache_path):
            with open(cache_path, "rb") as fo:
                X = pickle.load(fo)
            logger.info(f"Loaded KDDCUP99 dataset from cache with {X.shape[0]} samples and {X.shape[1]} features.")
            return X, None
        kddcup99 = fetch_openml('KDDCup99', version=1)
        X = kddcup99.data.select_dtypes(include=[np.number]).fillna(0).astype('float32').to_numpy()
        y = kddcup99.target
        logger.info(f"Loaded KDDCUP99 dataset with {X.shape[0]} samples and {X.shape[1]} features.")
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
    elif name == 'IMAGENET':
        # Load image embeddings using a pretrained ResNet50 from torchvision.
        # dataset_id can be used to pass the path to the ImageNet root (ImageNet/ILSVRC2012-style)
        cache_dir = os.path.join(".cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"imagenet_embeddings_{npoints}_{ndim}_{random_seed}.pickle")
        if os.path.isfile(cache_path):
            with open(cache_path, "rb") as fo:
                X, y = pickle.load(fo)
            logger.info(f"Loaded ImageNet embeddings from cache with {X.shape[0]} samples and {X.shape[1]} features.")
            return X, y

        imagenet_dir = None
        if dataset_id is not None and isinstance(dataset_id, str) and os.path.isdir(dataset_id):
            imagenet_dir = dataset_id
        else:
            imagenet_dir = os.getenv('IMAGENET_DIR')

        if imagenet_dir is None:
            raise RuntimeError("IMAGENET option requires a path to ImageNet images. Set `dataset_id` to the root path or set the IMAGENET_DIR environment variable.")

        # reproducibility
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        # transforms matching ResNet pretrained expectations
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset = tv_datasets.ImageFolder(imagenet_dir, transform=transform)
        total = len(dataset)
        if total == 0:
            raise RuntimeError(f"No images found in ImageNet directory: {imagenet_dir}")

        if npoints and npoints > 0:
            n_use = min(npoints, total)
            indices = list(range(n_use))
            dataset = Subset(dataset, indices)
        else:
            n_use = total

        batch_size = 256
        num_workers = min(4, (os.cpu_count() or 1))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if hasattr(models, 'ResNet50_Weights') else None)
        # strip final fc layer — get pooled features of size 2048
        feat_extractor = torch.nn.Sequential(*list(base_model.children())[:-1])
        feat_extractor.to(device)
        feat_extractor.eval()

        embeddings = []
        labels = []
        with torch.no_grad():
            for batch in loader:
                imgs, labs = batch
                imgs = imgs.to(device)
                feats = feat_extractor(imgs)
                feats = feats.view(feats.size(0), -1)
                embeddings.append(feats.cpu().numpy())
                labels.append(np.array(labs))

        X = np.vstack(embeddings).astype(np.float32)
        y = np.concatenate(labels).astype(np.int32)

        # Dimensionality reduction if requested
        if ndim and ndim > 0 and ndim < X.shape[1]:
            logger.info(f"Applying TruncatedSVD to reduce embeddings {X.shape[1]} -> {ndim}")
            svd = TruncatedSVD(n_components=ndim, random_state=random_seed)
            X = svd.fit_transform(X).astype(np.float32)

        # Cache
        with open(cache_path, "wb") as fo:
            pickle.dump((X, y), fo)
        logger.info(f"Saved ImageNet embeddings to cache: {cache_path}")
    else:
        logger.info(f"Using dataset from OpenML: {name}")
        # Assumes it is a openml / sklearn dataset
        lower_name = name.lower()
        cache_path = os.path.join(
            f"./.cache/{lower_name}.pickle"
        )
        if os.path.isfile(cache_path):
            with open(cache_path, "rb") as fo:
                X = pickle.load(fo)
            logger.info(f"Loaded {name} dataset from cache with {X.shape[0]} samples and {X.shape[1]} features.")
            return X, None
        # X = fetch_openml(name, version=1)
        if dataset_id is not None:
            # X = fetch_openml(data_id=dataset_id)
            dataset = openml.datasets.get_dataset(dataset_id)
        else:
            # X = fetch_openml(name)
            dataset = openml.datasets.get_dataset(name)
        X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)
        X = X.select_dtypes(include=[np.number]).fillna(0).astype('float32').to_numpy()
        y = y
        logger.info(f"Loaded {name} dataset with {X.shape[0]} samples and {X.shape[1]} features.")
        

    # Ensure that there isn't no duplicates nor NaNs
    # Remove them if any
    X = np.nan_to_num(X)
    _, unique_indices = np.unique(X, axis=0, return_index=True)
    X = X[unique_indices]
    if y is not None:
        y = y[unique_indices]

    # NaN check
    if np.isnan(X).any():
        logger.error("Dataset contains NaN values after cleaning. Replacing NaNs with zeros.")
        X = np.nan_to_num(X)

    with open(cache_path, "wb") as fo:
        pickle.dump(X, fo)
    
    return X, y

def load_dataset_knn(
    name,
    max_k=128,
    npoints=0,
    ndim=0,
    random_seed=0,
    return_brute_force_time=False
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
            return data[0][:,:max_k], data[1][:,:max_k], None

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

    brute_force_time = time.time()-init_t
    
    logger.info(f"Faiss knn search completed in {brute_force_time:.2f} seconds.")


    fo = open(knn_file, "wb")
    pickle.dump((distances,indices),fo)

    if return_brute_force_time:
        return distances, indices, brute_force_time
    
    return distances, indices
