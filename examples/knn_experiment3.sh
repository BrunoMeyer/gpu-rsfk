# DATASET_NAME="GOOGLE_NEWS300"
# DATASET_NAME="AMAZON_REVIEW_ELETRONICS"
# DATASET_NAME="MNIST"
# DATASET_NAME="ATSNE_MNIST"
# DATASET_NAME="ATSNE_IMAGENET"
DATASET_NAME="ARTIFICIAL_UNIFORM"



python3 knn_experiment3.py --test_rsfk --sanity_check --save_plot -d "$DATASET_NAME" -v 1 -k 32 --mntc 256 --mxtc 1024 -t 128 -e 0
# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d "$DATASET_NAME" -v 2 -k 32 --mntc 33 --mxtc 128 -t 1 2 4 8 16 32 64 128 256 384 512 640 768 896
# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d "$DATASET_NAME" -v 2 -k 32 --mntc 33 --mxtc 256 -e 0 -t 1 2 4 8 16 32 64 128 256 384 512 640 768 896
