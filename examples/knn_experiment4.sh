# DATASET_NAME="GOOGLE_NEWS300"
# DATASET_NAME="AMAZON_REVIEW_ELETRONICS"
# DATASET_NAME="MNIST"

# DATASET_NAME="MNIST_SKLEARN"
# DATASET_NAME="ATSNE_MNIST"
DATASET_NAME="ATSNE_IMAGENET"
# DATASET_NAME="GOOGLE_NEWS300"

# DATASET_NAME="ATSNE_MNIST"
# python3 knn_experiment4.py --test_rsfk --sanity_check --save_plot -d "$DATASET_NAME" -v 1 -k 32 --mntc 256 --mxtc 1024 -t 1 2 4 8 16 32 64 128 256 512

# DATASET_NAME="ATSNE_IMAGENET"
# python3 knn_experiment4.py --test_rsfk --sanity_check --save_plot -d "$DATASET_NAME" -v 1 -k 32 --mntc 256 --mxtc 1024 -t 1 2 4 8 16 32 64 128 256 512 -e 1

# DATASET_NAME="GOOGLE_NEWS300"
# python3 knn_experiment4.py --test_rsfk --sanity_check -d "$DATASET_NAME" -v 1 -k 32 --mntc 256 --mxtc 1024 -t 1 2 4 8 16 32 64 128 256 512 -e 1

# python3 knn_experiment4.py --test_annoy --sanity_check --save_plot -d "$DATASET_NAME" -v 1 -k 32
# python3 knn_experiment4.py --test_ivfflat --sanity_check --save_plot -d "$DATASET_NAME" -v 1 -k 32


# python3 knn_experiment4.py --skip_save --save_plot -d "$DATASET_NAME" -v 1 -k 32
# python3 knn_experiment4.py --skip_save -d "$DATASET_NAME" -v 1 -k 32
# python3 knn_experiment4.py --skip_save --test_ivfflat -d "$DATASET_NAME" -v 1 -k 32

# DATASET_NAME="ATSNE_IMAGENET"
# python3 knn_experiment4.py --test_rsfk --sanity_check -d "$DATASET_NAME" -v 1 -k 32 --mntc 256 --mxtc 1024 -t 1 2 4 8 16 32 64 128 256 512 -e 1 --skip_save
# python3 knn_experiment4.py -d "$DATASET_NAME" -v 1 -k 32 --mntc 256 --mxtc 1024 -t 1 2 4 8 16 32 64 128 256 512 -e 1 --skip_save


python3 knn_experiment4.py -d "$DATASET_NAME" -v 1 --skip_save --save_plot



# python3 knn_experiment4.py -d "$DATASET_NAME" -v 1 --skip_save --save_plot

# python3 knn_experiment4.py --test_ivfflat -d "$DATASET_NAME" -v 1 --save_plot

# python3 knn_experiment4.py --test_rsfk --sanity_check -d "$DATASET_NAME" -v 1 -k 32 --mntc 256 --mxtc 1024 -t 1 2 4 8 16 32 64 128 256 512 -e 1