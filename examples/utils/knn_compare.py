import os
import json
import matplotlib.pyplot as plt
import numpy as np

def get_nne_rate(h_indices, l_indices, random_state=0, max_k=32,
                 verbose=0):

    if verbose >= 2:
        print("Precision \t|\t Recall")
    
    if verbose >= 1:
        iterator_1 = tqdm(zip(h_indices, l_indices))
    else:
        iterator_1 = zip(h_indices, l_indices)

    total_T = 0
    for x,y in iterator_1:
        T = len(set(x).intersection(set(y)))
        total_T+=T

    N = len(h_indices)

    qnx = float(total_T)/(N * max_k)
    return qnx

    # rnx = ((N-1)*qnx-max_k)/(N-1-max_k)
    # return rnx

class KnnResult(object):
    def __init__(self, dir_path=".", experiment_name="exp"):
        self._dir_path = dir_path
        self._path = os.path.join(dir_path,"{}.json".format(experiment_name))
        self._experiment_name = experiment_name

        self.data = {}
        if os.path.exists(self._path):
            with open(self._path) as handle:
                self.data = json.loads(handle.read())

    def add_knn_result(self, dataset_name, K, knn_method_name, parameter_name, parameter_list,
                       quality_name, quality_list, time_list, save_after_add=True):
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

        if save_after_add:
            self.save()

    def save(self):
        if not os.path.exists(self._dir_path):
            os.makedirs(self._dir_path)

        with open(self._path, "w") as json_file:
            json.dump(self.data, json_file, indent=4)

    def plot(self, dataset_name, K, quality_name, dataX=None, dash_method=[]):
        fig, ax = plt.subplots(figsize=(16, 8))

        for knn_method_name in self.data[dataset_name][str(K)]:
            for parameter_name in self.data[dataset_name][str(K)][knn_method_name]:
                legend_name = str(knn_method_name)+" ({})".format(parameter_name)
                data = self.data[dataset_name][str(K)][knn_method_name][parameter_name]
                
                parameter_list = data[quality_name]["parameters"]
                quality_list = np.array(data[quality_name]["quality"])
                time_list = np.array(data[quality_name]["time"])

                if not dataX is None:
                    time_list = len(dataX)/time_list
                
                idx = np.argsort(quality_list)

                if knn_method_name in dash_method:
                    ax.plot(quality_list[idx], time_list[idx], ":", label=legend_name)
                else:
                    ax.plot(quality_list[idx], time_list[idx], label=legend_name)

        
        ax.legend()
        fig.savefig("{}.png".format(self._experiment_name))
