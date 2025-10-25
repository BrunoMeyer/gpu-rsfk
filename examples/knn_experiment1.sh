# DATASET_NAME="GOOGLE_NEWS300"
# DATASET_NAME="AMAZON_REVIEW_ELETRONICS"
# DATASET_NAME="MNIST"
# DATASET_NAME="ATSNE_MNIST"
DATASET_NAME="ATSNE_IMAGENET"

# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d AMAZON_REVIEW_ELETRONICS -v 2 --recall_eps_val 0.01 -k 8 >> knn_experiment1_recall_eps_001.txt
# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d AMAZON_REVIEW_ELETRONICS -v 2 --recall_eps_val 0.01 -k 8 --mntc 9 --mxtc 32 >> knn_experiment1_recall_eps_001.txt
# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d AMAZON_REVIEW_ELETRONICS -v 2 --recall_eps_val 0.01 -k 8 --mntc 9 --mxtc 64 >> knn_experiment1_recall_eps_001.txt
# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d AMAZON_REVIEW_ELETRONICS -v 2 --recall_eps_val 0.01 -k 8 --mntc 9 --mxtc 128 >> knn_experiment1_recall_eps_001.txt
# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d AMAZON_REVIEW_ELETRONICS -v 2 --recall_eps_val 0.01 -k 32 >> knn_experiment1_recall_eps_001.txt
# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d AMAZON_REVIEW_ELETRONICS -v 2 --recall_eps_val 0.01 -k 32 --mntc 33 --mxtc 128 >> knn_experiment1_recall_eps_001.txt
# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d AMAZON_REVIEW_ELETRONICS -v 2 --recall_eps_val 0.01 -k 32 --mntc 33 --mxtc 256 >> knn_experiment1_recall_eps_001.txt
# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d AMAZON_REVIEW_ELETRONICS -v 2 --recall_eps_val 0.01 -k 64 >> knn_experiment1_recall_eps_001.txt
# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d AMAZON_REVIEW_ELETRONICS -v 2 --recall_eps_val 0.01 -k 64 --mntc 65 --mxtc 256 >> knn_experiment1_recall_eps_001.txt

# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d AMAZON_REVIEW_ELETRONICS -v 2 --recall_eps_val 0.1 -k 8 >> knn_experiment1_recall_eps_01.txt
# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d AMAZON_REVIEW_ELETRONICS -v 2 --recall_eps_val 0.1 -k 8 --mntc 9 --mxtc 32 >> knn_experiment1_recall_eps_01.txt
# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d AMAZON_REVIEW_ELETRONICS -v 2 --recall_eps_val 0.1 -k 8 --mntc 9 --mxtc 64 >> knn_experiment1_recall_eps_01.txt
# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d AMAZON_REVIEW_ELETRONICS -v 2 --recall_eps_val 0.1 -k 8 --mntc 9 --mxtc 128 >> knn_experiment1_recall_eps_01.txt
# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d AMAZON_REVIEW_ELETRONICS -v 2 --recall_eps_val 0.1 -k 32 >> knn_experiment1_recall_eps_01.txt
# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d AMAZON_REVIEW_ELETRONICS -v 2 --recall_eps_val 0.1 -k 32 --mntc 33 --mxtc 128 >> knn_experiment1_recall_eps_01.txt
# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d AMAZON_REVIEW_ELETRONICS -v 2 --recall_eps_val 0.1 -k 32 --mntc 33 --mxtc 256 >> knn_experiment1_recall_eps_01.txt
# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d AMAZON_REVIEW_ELETRONICS -v 2 --recall_eps_val 0.1 -k 64 >> knn_experiment1_recall_eps_01.txt
# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d AMAZON_REVIEW_ELETRONICS -v 2 --recall_eps_val 0.1 -k 64 --mntc 65 --mxtc 256 >> knn_experiment1_recall_eps_01.txt

# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d AMAZON_REVIEW_ELETRONICS -v 2 -k 8 >> knn_experiment1_nne.txt
# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d AMAZON_REVIEW_ELETRONICS -v 2 -k 8 --mntc 9 --mxtc 32 >> knn_experiment1_nne.txt
# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d AMAZON_REVIEW_ELETRONICS -v 2 -k 8 --mntc 9 --mxtc 64 >> knn_experiment1_nne.txt
# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d AMAZON_REVIEW_ELETRONICS -v 2 -k 8 --mntc 9 --mxtc 128 >> knn_experiment1_nne.txt
# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d AMAZON_REVIEW_ELETRONICS -v 2 -k 32 >> knn_experiment1_nne.txt
# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d AMAZON_REVIEW_ELETRONICS -v 2 -k 32 --mntc 33 --mxtc 128 >> knn_experiment1_nne.txt
# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d AMAZON_REVIEW_ELETRONICS -v 2 -k 32 --mntc 33 --mxtc 256 >> knn_experiment1_nne.txt
# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d AMAZON_REVIEW_ELETRONICS -v 2 -k 64 -e 0 1 >> knn_experiment1_nne.txt
# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d AMAZON_REVIEW_ELETRONICS -v 2 -k 64 -e 0 1 --mntc 65 --mxtc 256 >> knn_experiment1_nne.txt
# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d AMAZON_REVIEW_ELETRONICS -v 2 -k 128 -e 0 -t 1 11 16 21 26 31 36 41 46 51 >> knn_experiment1_nne.txt
# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d AMAZON_REVIEW_ELETRONICS -v 2 -k 128 -e 0 -t 1 11 16 21 26 31 36 41 46 51 --mntc 8 --mxtc 128 >> knn_experiment1_nne.txt
# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d AMAZON_REVIEW_ELETRONICS -v 2 -k 128 -e 0 -t 1 11 16 21 26 31 36 41 46 51 --mntc 32 --mxtc 128 >> knn_experiment1_nne.txt
# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d AMAZON_REVIEW_ELETRONICS -v 2 -k 128 -e 0 -t 1 11 16 21 26 31 36 41 46 51 --mntc 32 --mxtc 256 >> knn_experiment1_nne.txt

# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d "$DATASET_NAME" -v 2 -k 8 --mntc 9 --mxtc 18
# # python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d "$DATASET_NAME" -v 2 -k 8 --mntc 9 --mxtc 32
# # python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d "$DATASET_NAME" -v 2 -k 8 --mntc 9 --mxtc 64
# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d "$DATASET_NAME" -v 2 -k 8 --mntc 9 --mxtc 128
# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d "$DATASET_NAME" -v 2 -k 32 --mntc 33 --mxtc 66
# # python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d "$DATASET_NAME" -v 2 -k 32 --mntc 33 --mxtc 128
# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d "$DATASET_NAME" -v 2 -k 32 --mntc 33 --mxtc 256
# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d "$DATASET_NAME" -v 2 -k 64 -e 0 1 --mntc 65 --mxtc 130
# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d "$DATASET_NAME" -v 2 -k 64 -e 0 1 --mntc 65 --mxtc 256
# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d "$DATASET_NAME" -v 2 -k 128 -e 0 -t 1 11 16 21 26 31 36 41 46 51 --mntc 129 --mxtc 258
# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d "$DATASET_NAME" -v 2 -k 128 -e 0 -t 1 11 16 21 26 31 36 41 46 51 --mntc 8 --mxtc 128
# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d "$DATASET_NAME" -v 2 -k 128 -e 0 -t 1 11 16 21 26 31 36 41 46 51 --mntc 32 --mxtc 128
# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d "$DATASET_NAME" -v 2 -k 128 -e 0 -t 1 11 16 21 26 31 36 41 46 51 --mntc 32 --mxtc 256


# python3 knn_experiment1.py --save_plot --skip_save -d AMAZON_REVIEW_ELETRONICS -v 2 -k 8
# python3 knn_experiment1.py --save_plot --skip_save -d AMAZON_REVIEW_ELETRONICS -v 2 -k 32
# python3 knn_experiment1.py --save_plot --skip_save -d AMAZON_REVIEW_ELETRONICS -v 2 -k 64
# python3 knn_experiment1.py --save_plot --skip_save -d AMAZON_REVIEW_ELETRONICS -v 2 -k 128

# python3 knn_experiment1.py --save_plot -d AMAZON_REVIEW_ELETRONICS -v 2 --recall_eps_val 0.0 -k 8 --test_ivfflat
# python3 knn_experiment1.py --save_plot -d AMAZON_REVIEW_ELETRONICS -v 2 --recall_eps_val 0.0 -k 32 --test_ivfflat
# python3 knn_experiment1.py --save_plot -d AMAZON_REVIEW_ELETRONICS -v 2 --recall_eps_val 0.0 -k 64 --test_ivfflat
# python3 knn_experiment1.py --save_plot -d AMAZON_REVIEW_ELETRONICS -v 2 --recall_eps_val 0.0 -k 128 --test_ivfflat

# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d "$DATASET_NAME" -v 2 -k 32 --mntc 33 --mxtc 66 -t 1 2 4 8 16 32 64 128 256
# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d "$DATASET_NAME" -v 2 -k 32 --mntc 33 --mxtc 128 -t 1 2 4 8 16 32 64 128 256
# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d "$DATASET_NAME" -v 2 -k 32 --mntc 33 --mxtc 256 -t 1 2 4 8 16 32 64 128 256
# python3 knn_experiment1.py --save_plot -d "$DATASET_NAME" -v 2 --recall_eps_val 0.0 -k 8 --test_ivfflat --test_flatl2
# python3 knn_experiment1.py --save_plot -d "$DATASET_NAME" -v 2 --recall_eps_val 0.0 -k 32 --test_ivfflat --test_flatl2
# python3 knn_experiment1.py --save_plot -d "$DATASET_NAME" -v 2 --recall_eps_val 0.0 -k 64 --test_ivfflat --test_flatl2
# python3 knn_experiment1.py --save_plot -d "$DATASET_NAME" -v 2 --recall_eps_val 0.0 -k 128 --test_ivfflat --test_flatl2

# python3 knn_experiment1.py --save_plot --skip_save -d "$DATASET_NAME" -v 2 -k 8
# python3 knn_experiment1.py --save_plot --skip_save -d "$DATASET_NAME" -v 2 -k 32
# python3 knn_experiment1.py --save_plot --skip_save -d "$DATASET_NAME" -v 2 -k 64
# python3 knn_experiment1.py --save_plot --skip_save -d "$DATASET_NAME" -v 2 -k 128


python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d "$DATASET_NAME" -v 2 -k 128 --mntc 129 --mxtc 300 -t 1 2 4 8 16 32 64 128 256 384 512 640 768 896
# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d "$DATASET_NAME" -v 2 -k 32 --mntc 33 --mxtc 128 -t 1 2 4 8 16 32 64 128 256 384 512 640 768 896
# python3 knn_experiment1.py --test_rsfk --sanity_check --save_plot -d "$DATASET_NAME" -v 2 -k 32 --mntc 33 --mxtc 256 -e 0 -t 1 2 4 8 16 32 64 128 256 384 512 640 768 896
