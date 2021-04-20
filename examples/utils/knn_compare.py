import os
import json
import matplotlib.pyplot as plt
import numpy as np
import warnings
from cycler import cycler

import matplotlib

def get_projection_val(serie_x, serie_y, value):
    serie_sorted_x = serie_x[np.argsort(serie_x)]
    serie_sorted_y = serie_y[np.argsort(serie_x)]

    idx = np.searchsorted(serie_sorted_x, value)
    # assert idx!=0

    # if idx>=len(serie_x):
    # if idx==0 or idx>=len(serie_x):
    # if idx==0:
        # return None
    if idx==0:
        idx = 1
    if idx>=len(serie_x):
        idx = len(serie_x)-1

    
    diff = serie_sorted_x[idx] - serie_sorted_x[idx-1]
    diff2 = value - serie_sorted_x[idx-1]
    r = diff2/diff

    diff_y = serie_sorted_y[idx] - serie_sorted_y[idx-1]

    return serie_sorted_y[idx-1] + (r*diff_y)
    # return (serie_sorted_y[idx-1]+serie_sorted_y[idx])/2



def get_projection(serie_x, serie_y, values):
    return np.array([get_projection_val(serie_x, serie_y, v) for v in values])

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
        
    def plot(self, dataset_list, K, quality_name, dataX=None, dash_method=[],
             fig_name=None, ignore_outliers=True, baseline=None, method_list=None):
        
        font = {
            # 'family' : 'normal',
            # 'weight' : 'bold',
            'size'   : 19
        }

        matplotlib.rc('xtick', labelsize=22) 
        matplotlib.rc('ytick', labelsize=22)


        plt.rc('font', **font)

        assert (type(dataset_list) == str) or (type(dataset_list) == list)
        if type(dataset_list) is str:
            dataset_list = [dataset_list]


        
        if fig_name is None:
            fig_name = "{}_{}".format(quality_name,"_".join(dataset_list))+str(self._experiment_name)+"_K{}".format(K)

        

        fig, ax = plt.subplots(figsize=(10, 6))

        if not baseline is None:
            ax2 = ax.twinx()
            ivfflat_x = None
            ivfflat_y = None
            plt_colors_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            

        min_x = np.inf
        max_x = 0
        min_y = np.inf
        max_y = 0
        min_y2 = np.inf
        max_y2 = 0
        non_baseline_count = 0

        
        for dataset_name in dataset_list:
            if method_list is None: 
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
                    if len(dataset_list) > 1:
                        legend_name = "({}) ".format(dataset_name) + legend_name
                    
                    if knn_method_name == "IVFFLAT":
                        legend_name = legend_name.replace("IVFFLAT", "FAISS IVFFLAT")

                    legend_name = legend_name.replace("GOOGLE_NEWS300", "GoogleNews300")
                    legend_name = legend_name.replace("AMAZON_REVIEW_ELETRONICS", "Amazon Electronics")
                    legend_name = legend_name.replace("LUCID_INCEPTION", "Lucid Inception")
                    legend_name = legend_name.replace("ATSNE_MNIST", "MNIST")
                    legend_name = legend_name.replace("ATSNE_IMAGENET", "Imagenet")
                    legend_name = legend_name.replace("ATSNE_CIFAR", "CIFAR")

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

                    min_x = min(min_x, min(quality_list[idx]))
                    min_y = min(min_y, min(time_list[idx]))

                    max_x = max(max_x, max(quality_list[idx]))
                    max_y = max(max_y, max(time_list[idx]))
                    
                    if knn_method_name == baseline:
                        ivfflat_x = quality_list[idx]
                        ivfflat_y = time_list[idx]
            
            

            if (not baseline is None) and (not ivfflat_x is None):
                ax2.set_prop_cycle(cycler('color', plt_colors_cycle[non_baseline_count+1:]))


                for knn_method_name in method_list:
                    for parameter_name in self.data[dataset_name][str(K)][knn_method_name]:
                        if knn_method_name =="IVFFLAT" or knn_method_name in dash_method:
                            continue
                        
                        # legend_name = str(knn_method_name)+" ({})".format(parameter_name)
                        legend_name = str(knn_method_name)
                        if len(dataset_list) > 1:
                            legend_name = "({}) ".format(dataset_name) + legend_name

                        legend_name = legend_name.replace("GOOGLE_NEWS300", "GoogleNews300")
                        legend_name = legend_name.replace("AMAZON_REVIEW_ELETRONICS", "Amazon Electronics")
                        legend_name = legend_name.replace("LUCID_INCEPTION", "Lucid Inception")
                        legend_name = legend_name.replace("ATSNE_MNIST", "MNIST")
                        legend_name = legend_name.replace("ATSNE_IMAGENET", "Imagenet")
                        legend_name = legend_name.replace("ATSNE_CIFAR", "CIFAR")

                        data = self.data[dataset_name][str(K)][knn_method_name][parameter_name]
                        if not quality_name in data:
                            continue
                        
                        parameter_list = data[quality_name]["parameters"]
                        quality_list = np.array(data[quality_name]["quality"])
                        time_list = np.array(data[quality_name]["time"])

                        if not dataX is None:
                            time_list = len(dataX)/time_list
                        
                        idx = np.argsort(quality_list)

                        proj = get_projection(ivfflat_x, ivfflat_y, quality_list[idx])
                        fill_time = time_list[idx][proj != None]
                        fill_proj = proj[proj != None]
                        fill_quality = quality_list[idx][proj != None]

                        speedup = fill_time/fill_proj
                        # print(legend_name, ivfflat_x, quality_list[idx])
                        # print(proj)
                        # print(legend_name, proj, fill_time, fill_proj)
                        ax2.plot(fill_quality, speedup, ":", label="Speedup " + legend_name)

                        min_y2 = min(min_y2, min(speedup))

                        max_y2 = max(max_y2, max(speedup))
                        
                    # ax2.ticklabel_format(useOffset=False, style='plain')
                    # ax2.yaxis.get_major_formatter().set_scientific(False)
                    # ax2.legend()
                    # ax2.legend(loc=0)

                non_baseline_count+=len(method_list)
                
                ax2.set_ylabel('Speedup')
                ax2.set_ylim(1)
                major_ticks = np.arange(1, max_y2, 2)
                minor_ticks = np.arange(1, max_y2, 1)
                ax2.set_yticks(major_ticks)
                ax2.set_yticks(minor_ticks, minor=True)
                ax2.grid(which='minor', alpha=0.2)
                ax2.grid(which='major', alpha=0.5)

        # ax.legend()
        fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes, framealpha=0.5, prop={'size': 14})
        # fig.legend()
        # plt.legend()

        # major_ticks = np.arange(min_x, max_x, 1)
        # minor_ticks = np.arange(min_x, max_x, 0.5)
        # ax.set_yticks(major_ticks)
        # ax.set_yticks(minor_ticks, minor=True)

        # major_ticks = np.arange(1, max_y2, 1)
        # minor_ticks = np.arange(0, max_y2, 0.5)
        # ax.set_xticks(major_ticks)
        # ax.set_xticks(minor_ticks, minor=True)
        
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)


        # And a corresponding grid
        ax.grid(which='both')

        # Or if you want different settings for the grids:
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)



        ax.set_xlabel('K-NNG Accuracy')
        ax.set_ylabel('Average of points treated per second')
        # ax1.set_title('a sine wave')
        fig_title  = "{}-Nearest Neighbors".format(K)
        ax.set_yscale('log')

        # TODO: Adicionar legenda no canto esquerdo inferior
        # TODO: Speedup projetado como anotação nos ticks (pontos)


        fig_title = "{} {}-Nearest Neighbors".format(" | ".join(dataset_list), K)
        fig_title = fig_title.replace("GOOGLE_NEWS300", "GoogleNews300")
        fig_title = fig_title.replace("AMAZON_REVIEW_ELETRONICS", "Amazon Electronics")
        fig_title = fig_title.replace("LUCID_INCEPTION", "Lucid Inception")
        fig_title = fig_title.replace("ATSNE_MNIST", "MNIST")
        fig_title = fig_title.replace("ATSNE_IMAGENET", "Imagenet")
        fig_title = fig_title.replace("ATSNE_CIFAR", "CIFAR")

        if dataset_name == "GOOGLE_NEWS300":
            if quality_name == "dist_mean":
                plt.xlim(3.25,5.5)
            
        # if dataset_name == "AMAZON_REVIEW_ELETRONICS":
        #     fig_title = "Amazon Electronics {}-Nearest Neighbors".format(K)
        # if dataset_name == "LUCID_INCEPTION":
        #     fig_title = "Lucid Inception {}-Nearest Neighbors".format(K)
        # if dataset_name == "MNIST":
        #     fig_title = "MNIST {}-Nearest Neighbors".format(K)
        # if dataset_name == "CIFAR":
        #     fig_title = "CIFAR {}-Nearest Neighbors".format(K)
        
        # ax.set_title(fig_title)
        
        fig.savefig("{}.pdf".format(fig_name))
