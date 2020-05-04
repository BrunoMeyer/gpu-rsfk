import os
import json
import matplotlib.pyplot as plt
import numpy as np
import warnings

def get_nne_rate(real_indices, indices, real_dist=None, dist=None,
                 random_state=0, max_k=32, verbose=0):

    if verbose >= 2:
        print("Precision \t|\t Recall")
    
    if verbose >= 1:
        iterator_1 = tqdm(zip(real_indices, indices))
    else:
        iterator_1 = zip(real_indices, indices)

    total_T = 0
    for x,y in iterator_1:
        T = len(set(x).intersection(set(y)))
        total_T+=T

    N = len(real_indices)

    qnx = float(total_T)/(N * max_k)
    return qnx

    # rnx = ((N-1)*qnx-max_k)/(N-1-max_k)
    # return rnx


def create_recall_eps(eps):
    def get_recall_eps(real_indices, indices, real_dist, dist,
                    random_state=0, max_k=32, verbose=0):

        if verbose >= 2:
            print("Precision \t|\t Recall")
        
        if verbose >= 1:
            iterator_1 = tqdm(zip(real_indices, indices, real_dist, dist))
        else:
            iterator_1 = zip(real_indices, indices, real_dist, dist)

        total_T = 0
        for x,y,dx,dy in iterator_1:
            limiar_dist = np.max(dx)*(1.0+eps)
            T = np.sum(dy <= limiar_dist)
            total_T+=T

        N = len(real_indices)

        qnx = float(total_T)/(N * max_k)
        return qnx

    return get_recall_eps

class KnnResult(object):
    def __init__(self, dir_path=".", experiment_name="exp",
                 save_after_add=False):
        self._dir_path = dir_path
        self._path = os.path.join(dir_path,"{}.json".format(experiment_name))
        self._experiment_name = experiment_name
        self.save_after_add = save_after_add
        
        self.data = {}
        if os.path.exists(self._path):
            with open(self._path) as handle:
                self.data = json.loads(handle.read())

    def add_knn_result(self, dataset_name, K, knn_method_name, parameter_name, parameter_list,
                       quality_name, quality_list, time_list):
        obj = self.data
        
        if not dataset_name in obj:
            obj[dataset_name] = {}

        obj = self.data[dataset_name]

        if not str(K) in obj:
            obj[str(K)] = {}
        obj = obj[str(K)]

        if not knn_method_name in obj:
            obj[knn_method_name] = {}
        obj = obj[knn_method_name]

        if not parameter_name in obj:
            obj[parameter_name] = {}
        obj = obj[parameter_name]

        if not quality_name in obj:
            obj[quality_name] = {}

        obj[quality_name] = {
            "parameters": parameter_list,
            "quality": quality_list,
            "time": time_list,
        }

        if self.save_after_add:
            self.save()

    def save(self):
        if not os.path.exists(self._dir_path):
            os.makedirs(self._dir_path)

        with open(self._path, "w") as json_file:
            json.dump(self.data, json_file, indent=4)

    def remove_method_by_name(self, name):
        new_data = {}
        for dn in self.data:
            if not dn in new_data:
                new_data[dn] = {}
            for k in self.data[dn]:
                if not k in new_data[dn]:
                    new_data[dn][k] = {}
                for method_name in self.data[dn][k]:
                    if not name in method_name:
                        new_data[dn][k][method_name] = self.data[dn][k][method_name]
        self.data = new_data
        
    def plot(self, dataset_name, K, quality_name, dataX=None, dash_method=[],
             fig_name=None, ignore_outliers=True):
        if fig_name is None:
            fig_name = "{}_{}".format(quality_name,dataset_name)+str(self._experiment_name)+"_K{}".format(K)
        
        fig, ax = plt.subplots(figsize=(8, 8))

        method_list = list(self.data[dataset_name][str(K)].keys())
        if "FLATL2" in method_list:
            method_list.remove("FLATL2")
            method_list = ["FLATL2"]+method_list
        if "IVFFLAT" in method_list:
            method_list.remove("IVFFLAT")
            method_list = ["IVFFLAT"]+method_list
            
        for knn_method_name in method_list:
            for parameter_name in self.data[dataset_name][str(K)][knn_method_name]:
                # legend_name = str(knn_method_name)+" ({})".format(parameter_name)
                legend_name = str(knn_method_name)
                legend_name = legend_name.replace("RPFK", "RSFK")
                data = self.data[dataset_name][str(K)][knn_method_name][parameter_name]
                if not quality_name in data:
                    continue
                 
                parameter_list = data[quality_name]["parameters"]
                quality_list = np.array(data[quality_name]["quality"])
                time_list = np.array(data[quality_name]["time"])

                if not dataX is None:
                    time_list = len(dataX)/time_list
                
                idx = np.argsort(quality_list)

                if knn_method_name in dash_method:
                    ax.plot(quality_list[idx], time_list[idx], ":|", label=legend_name)
                else:
                    ax.plot(quality_list[idx], time_list[idx], "-|", label=legend_name)

        
        ax.legend()
        ax.set_xlabel('KNN graph Accuracy')
        ax.set_ylabel('Average of points treated per second')
        # ax1.set_title('a sine wave')
        fig_title  = "{}-Nearest Neighbors".format(K)
        plt.yscale('log')
        if dataset_name == "GOOGLE_NEWS300":
            if quality_name == "dist_mean":
                plt.xlim(3.25,5.5)
            fig_title = "GoogleNews300 {}-Nearest Neighbors".format(K)
        if dataset_name == "AMAZON_REVIEW_ELETRONICS":
            fig_title = "Amazon Electronics {}-Nearest Neighbors".format(K)
        if dataset_name == "LUCID_INCEPTION":
            fig_title = "Lucid Inception {}-Nearest Neighbors".format(K)
        if dataset_name == "MNIST":
            fig_title = "MNIST {}-Nearest Neighbors".format(K)
        if dataset_name == "CIFAR":
            fig_title = "CIFAR {}-Nearest Neighbors".format(K)
        
        ax.set_title(fig_title)
        
        fig.savefig("{}.pdf".format(fig_name))
