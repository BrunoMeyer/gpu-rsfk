import pickle
import numpy as np
from sklearn.datasets import load_digits, load_iris
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import fetch_openml
import faiss
import openml

import time

import logging

import kagglehub
from gensim.models import KeyedVectors



logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

KNN_CACHE_DIR = "./.cache/knn/"
ARTIFICIAL_DATASET_DIR_CACHE = "./.cache/artificial_datasets/"
DATASET_PATH = os.environ.get("DATASET_PATH", "./data/datasets/")

# https://github.com/ZJULearning/AtSNE?tab=readme-ov-file
ATSNE_DATASETS = {
    'CIFAR10': {
        'dim': 1024,
        'npoints': 60000,
        'ncats': 10,
        'data_txt': 'http://downloads.zjulearning.org.cn/atsne/cifar10_data.txt',
        'data_fvecs': 'http://downloads.zjulearning.org.cn/atsne/cifar10_data.txt.fvecs',
        'label_txt': 'http://downloads.zjulearning.org.cn/atsne/cifar10_label.txt',
        'label_ivecs': 'http://downloads.zjulearning.org.cn/atsne/cifar10_label.txt.ivecs',
    },
    'CIFAR100': {
        'dim': 1024,
        'npoints': 60000,
        'ncats': 100,
        'data_txt': 'http://downloads.zjulearning.org.cn/atsne/cifar100_data.txt',
        'data_fvecs': 'http://downloads.zjulearning.org.cn/atsne/cifar100_data.txt.fvecs',
        'label_txt': 'http://downloads.zjulearning.org.cn/atsne/cifar100_label.txt',
        'label_ivecs': 'http://downloads.zjulearning.org.cn/atsne/cifar100_label.txt.ivecs',
    },
    'MNIST': {
        'dim': 784,
        'npoints': 70000,
        'ncats': 10,
        'data_txt': 'http://downloads.zjulearning.org.cn/atsne/mnist_vec784D_data.txt',
        'data_fvecs': 'http://downloads.zjulearning.org.cn/atsne/mnist_vec784D_data.txt.fvecs',
        'label_txt': 'http://downloads.zjulearning.org.cn/atsne/mnist_vec784D_label.txt',
        'label_ivecs': 'http://downloads.zjulearning.org.cn/atsne/mnist_vec784D_label.txt.ivecs',
    },
    'Fashion-MNIST': {
        'dim': 784,
        'npoints': 70000,
        'ncats': 10,
        'data_txt': 'http://downloads.zjulearning.org.cn/atsne/fashion_mnist_data.txt',
        'data_fvecs': 'http://downloads.zjulearning.org.cn/atsne/fashion_mnist_data.txt.fvecs',
        'label_txt': 'http://downloads.zjulearning.org.cn/atsne/fashion_mnist_label.txt',
        'label_ivecs': 'http://downloads.zjulearning.org.cn/atsne/fashion_mnist_label.txt.ivecs',
    },
    'AG’s News': {
        'dim': 100,
        'npoints': 120000,
        'ncats': 4,
        'data_txt': 'http://downloads.zjulearning.org.cn/atsne/agnews_data.txt',
        'data_fvecs': 'http://downloads.zjulearning.org.cn/atsne/agnews_data.txt.fvecs',
        'label_txt': 'http://downloads.zjulearning.org.cn/atsne/agnews_label.txt',
        'label_ivecs': 'http://downloads.zjulearning.org.cn/atsne/agnews_label.txt.ivecs',
    },
    'DBPedia': {
        'dim': 100,
        'npoints': 560000,
        'ncats': 14,
        'data_txt': 'http://downloads.zjulearning.org.cn/atsne/dbpedia_data.txt',
        'data_fvecs': 'http://downloads.zjulearning.org.cn/atsne/dbpedia_data.txt.fvecs',
        'label_txt': 'http://downloads.zjulearning.org.cn/atsne/dbpedia_label.txt',
        'label_ivecs': 'http://downloads.zjulearning.org.cn/atsne/dbpedia_label.txt.ivecs',
    },
    'ImageNet': {
        'dim': 128,
        'npoints': 1281167,
        'ncats': 1000,
        'data_txt': 'http://downloads.zjulearning.org.cn/atsne/imagenet_data.txt',
        'data_fvecs': 'http://downloads.zjulearning.org.cn/atsne/imagenet_data.txt.fvecs',
        'label_txt': 'http://downloads.zjulearning.org.cn/atsne/imagenet_label.txt',
        'label_ivecs': 'http://downloads.zjulearning.org.cn/atsne/imagenet_label.txt.ivecs',
    },
    'Yahoo': {
        'dim': 100,
        'npoints': 1400000,
        'ncats': 10,
        'data_txt': 'http://downloads.zjulearning.org.cn/atsne/yahoo_data.txt',
        'data_fvecs': 'http://downloads.zjulearning.org.cn/atsne/yahoo_data.txt.fvecs',
        'label_txt': 'http://downloads.zjulearning.org.cn/atsne/yahoo_label.txt',
        'label_ivecs': 'http://downloads.zjulearning.org.cn/atsne/yahoo_label.txt.ivecs',
    },
    'Crawl': {
        'dim': 300,
        'npoints': 200000,
        'ncats': 10,
        'data_txt': 'http://downloads.zjulearning.org.cn/atsne/crawl_data.txt',
        'data_fvecs': 'http://downloads.zjulearning.org.cn/atsne/crawl_data.txt.fvecs',
        'label_txt': 'http://downloads.zjulearning.org.cn/atsne/crawl_label.txt',
        'label_ivecs': 'http://downloads.zjulearning.org.cn/atsne/crawl_label.txt.ivecs',
    },
    'Amazon3M': {
        'dim': 100,
        'npoints': 3000000,
        'ncats': 5,
        'data_txt': 'http://downloads.zjulearning.org.cn/atsne/amazon_data.txt',
        'data_fvecs': 'http://downloads.zjulearning.org.cn/atsne/amazon_data.txt.fvecs',
        'label_txt': 'http://downloads.zjulearning.org.cn/atsne/amazon_label.txt',
        'label_ivecs': 'http://downloads.zjulearning.org.cn/atsne/amazon_label.txt.ivecs',
    },
    'Amazon20M': {
        'dim': 96,
        'npoints': 19531329,
        'ncats': 5,
        'data_txt': 'http://downloads.zjulearning.org.cn/atsne/amazon_reviews_us_Books_data.txt',
        'data_fvecs': 'http://downloads.zjulearning.org.cn/atsne/amazon_reviews_us_Books_data.txt.fvecs',
        'label_txt': 'http://downloads.zjulearning.org.cn/atsne/amazon_reviews_us_Books_label.txt',
        'label_ivecs': 'http://downloads.zjulearning.org.cn/atsne/amazon_reviews_us_Books_label.txt.ivecs',
    },
}

def read_fvecs(filepath, name):
    with open(filepath, 'rb') as f:
        # Read dimensionality
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]

        # Calculate total number of floats and read vector data
        # (file size - 4 bytes for dim) / 4 bytes per float
        num_vectors = (f.seek(0, 2) - 4) // (dim * 4) 
        f.seek(4) # Go back after reading dim
        
        vectors = np.fromfile(f, dtype=np.float32, count=dim * num_vectors)
        
        # Reshape into a 2D array
        vectors = vectors.reshape(-1, dim)
        return vectors

def read_ivecs(fname, name):
    with open(fname, 'rb') as f:
        # Read dimensionality
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]

        # Calculate total number of integers and read vector data
        # (file size - 4 bytes for dim) / 4 bytes per int
        num_vectors = (f.seek(0, 2) - 4) // (dim * 4) 
        f.seek(4) # Go back after reading dim
        
        vectors = np.fromfile(f, dtype=np.int32, count=dim * num_vectors)
        
        # Reshape into a 2D array
        vectors = vectors.reshape(-1, dim)
        return vectors

def load_atsne_dataset(name, download_path=DATASET_PATH):
    '''
    Load dataset from AtSNE repository (download if needed)
    '''
    # Ensure download_path exists
    os.makedirs(download_path, exist_ok=True)
    if name not in ATSNE_DATASETS:
        raise ValueError(f"Dataset {name} not found in AtSNE datasets.")
    dataset_info = ATSNE_DATASETS[name]
    data_fvecs_url = dataset_info['data_fvecs']
    label_ivecs_url = dataset_info['label_ivecs']
    data_fvecs_path = os.path.join(download_path, f"{name}_data.fvecs")
    label_ivecs_path = os.path.join(download_path, f"{name}_label.ivecs")

    # Download if not exists
    if not os.path.isfile(data_fvecs_path):
        import urllib.request
        logger.info(f"Downloading {name} data fvecs from {data_fvecs_url}...")
        urllib.request.urlretrieve(data_fvecs_url, data_fvecs_path)
    if not os.path.isfile(label_ivecs_path):
        import urllib.request
        logger.info(f"Downloading {name} label ivecs from {label_ivecs_url}...")
        urllib.request.urlretrieve(label_ivecs_url, label_ivecs_path)
    # Load fvecs and ivecs
    X = read_fvecs(data_fvecs_path, name)
    y = read_ivecs(label_ivecs_path, name).flatten()
    return X, y

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
        X = np.random.rand(n_samples, n_features).astype('float32')
        y = None
        logger.info(f"Created artificial uniform dataset with {X.shape[0]} samples and {X.shape[1]} features.")
        with open(cache_path, "wb") as fo:
            pickle.dump(X, fo)

    elif name == 'IMAGENET':
        # ImageNet embeddings (128D) using pretrained ResNet + PCA
        cache_path = os.path.join("./.cache/imagenet_128.pickle")
        if os.path.isfile(cache_path):
            with open(cache_path, "rb") as fo:
                X = pickle.load(fo)
            logger.info(f"Loaded IMAGENET (128D embeddings) dataset from cache with "
                        f"{X.shape[0]} samples and {X.shape[1]} features.")
            return X, None

        import torch
        from torch.utils.data import DataLoader, Subset
        import torchvision
        from torchvision import transforms
        from sklearn.decomposition import PCA

        # Where your ImageNet-like folder lives
        # Expecting structure: root/class_x/xxx.png etc.
        data_root = os.environ.get("IMAGENET_ROOT", "./data/imagenet")
        # Ensure data_root exists
        

        if not os.path.isdir(data_root):
            print(f"[WARNING] IMAGENET_ROOT directory {data_root} does not exist.")

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # More generic than torchvision.datasets.ImageNet: works with any ImageNet-style folder
        dataset = torchvision.datasets.ImageFolder(root=data_root, transform=transform)

        logger.info(f"Found {len(dataset)} ImageNet images in {data_root}.")

        if npoints > 0 and npoints < len(dataset):
            rng = np.random.default_rng(random_seed)
            subset_indices = rng.choice(len(dataset), size=npoints, replace=False)
            dataset = Subset(dataset, subset_indices)
            logger.info(f"Subsampled ImageNet to {len(dataset)} images (npoints={npoints}).")

        batch_size = 64
        num_workers = 4
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Pretrained ResNet-50 as feature extractor (2048D)
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        base_model = torchvision.models.resnet50(weights=weights)
        # Replace final FC layer with identity so output is penultimate feature (2048D)
        base_model.fc = torch.nn.Identity()
        base_model = base_model.to(device)
        base_model.eval()

        logger.info("Extracting 2048D ResNet features from ImageNet images...")
        feats = []
        with torch.no_grad():
            for i, (images, _) in enumerate(loader):
                images = images.to(device, non_blocking=True)
                out = base_model(images)
                feats.append(out.cpu().numpy())
                if (i + 1) % 50 == 0:
                    logger.info(f"Processed { (i+1) * batch_size } images...")

        feats = np.concatenate(feats, axis=0).astype('float32')
        logger.info(f"Extracted raw features with shape {feats.shape}.")

        # Reduce to 128 dimensions via PCA
        logger.info("Fitting PCA to reduce ImageNet features to 128 dimensions...")
        pca = PCA(n_components=128, random_state=random_seed)
        X = pca.fit_transform(feats).astype('float32')
        y = None
        logger.info(f"ImageNet embeddings shape after PCA: {X.shape}.")

    elif name.upper() in ('GOOGLENEWS300', 'GOOGLENEWS'):
        # GoogleNews word2vec embeddings (300D)
        cache_path = os.path.join("./.cache/googlenews300.pickle")
        if os.path.isfile(cache_path):
            with open(cache_path, "rb") as fo:
                X = pickle.load(fo)
            logger.info(f"Loaded GoogleNews300 dataset from cache with "
                        f"{X.shape[0]} samples and {X.shape[1]} features.")
            return X, None

        googlenews_path = kagglehub.dataset_download("leadbest/googlenewsvectorsnegative300")

        googlenews_path = os.path.join(
            googlenews_path,
            "GoogleNews-vectors-negative300.bin"
        )
        if not os.path.isfile(googlenews_path):
            raise RuntimeError(
                f"GoogleNews300 binary not found at {googlenews_path}. "
                "Set GOOGLENEWS_PATH to the path of GoogleNews-vectors-negative300.bin."
            )

        logger.info(f"Loading GoogleNews300 word2vec model from {googlenews_path} "
                    "(this may take a while)...")
        kv = KeyedVectors.load_word2vec_format(googlenews_path, binary=True)

        # kv.vectors is shape (n_words, 300)
        vectors = kv.vectors.astype('float32')
        logger.info(f"Loaded GoogleNews word vectors with shape {vectors.shape}.")

        # Optional subsampling
        if npoints > 0 and npoints < vectors.shape[0]:
            rng = np.random.default_rng(random_seed)
            idx = rng.choice(vectors.shape[0], size=npoints, replace=False)
            X = vectors[idx]
            logger.info(f"Subsampled GoogleNews300 to {X.shape[0]} vectors (npoints={npoints}).")
        else:
            X = vectors

        y = None
        logger.info(f"Final GoogleNews300 dataset shape: {X.shape}.")

    elif name.upper().startswith('ATSNE_'):
        atsne_name = name[6:]  # Remove 'ATSNE_' prefix
        logger.info(f"Using dataset from AtSNE repository: {atsne_name}")
        X, y = load_atsne_dataset(atsne_name)
        logger.info(f"Loaded {atsne_name} dataset with {X.shape[0]} samples and {X.shape[1]} features.")
        return X, y
    else:
        logger.info(f"Using dataset from OpenML: {name}")
        # Assumes it is an openml / sklearn dataset
        lower_name = name.lower()
        cache_path = os.path.join(
            f"./.cache/{lower_name}.pickle"
        )
        if os.path.isfile(cache_path):
            with open(cache_path, "rb") as fo:
                X = pickle.load(fo)
            logger.info(f"Loaded {name} dataset from cache with {X.shape[0]} samples and {X.shape[1]} features.")
            return X, None
        if dataset_id is not None:
            dataset = openml.datasets.get_dataset(dataset_id)
        else:
            dataset = openml.datasets.get_dataset(name)
        X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)
        X = X.select_dtypes(include=[np.number]).fillna(0).astype('float32').to_numpy()
        y = y
        logger.info(f"Loaded {name} dataset with {X.shape[0]} samples and {X.shape[1]} features.")
        
    # ===== Post-processing common to all datasets (except those returned early from cache) =====
    # Ensure that there isn't no duplicates nor NaNs
    # Remove them if any
    X = np.nan_to_num(X)
    _, unique_indices = np.unique(X, axis=0, return_index=True)
    X = X[unique_indices]
    if 'y' in locals() and y is not None:
        y = np.array(y)[unique_indices]

    # NaN check
    if np.isnan(X).any():
        logger.error("Dataset contains NaN values after cleaning. Replacing NaNs with zeros.")
        X = np.nan_to_num(X)

    # Make sure cache_path exists for branches that should be cached
    if 'cache_path' in locals():
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
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

    nq, d = dataX.shape

    init_t = time.time()
    index = faiss.IndexFlatL2(d)
    index = faiss.index_cpu_to_gpu(res, 0, index)

    # here we specify METRIC_L2, by default it performs inner-product search
    logger.info("Training faiss index...")
    index.train(dataX)
    logger.info("Adding vectors to faiss index...")
    index.add(dataX)                  # add may be a bit slower as well
    distances, indices = index.search(dataX, max_k)     # actual search

    brute_force_time = time.time()-init_t
    
    logger.info(f"Faiss knn search completed in {brute_force_time:.2f} seconds.")

    with open(knn_file, "wb") as fo:
        pickle.dump((distances,indices),fo)

    if return_brute_force_time:
        return distances, indices, brute_force_time
    
    return distances, indices
