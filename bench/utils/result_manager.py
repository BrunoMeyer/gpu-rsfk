import os
import json
import matplotlib.pyplot as plt
import numpy as np
import warnings
from cycler import cycler

import matplotlib
from PIL import ImageColor

import pandas as pd
import time

from utils.metrics import get_projection, get_nne_rate
from utils.datasets import load_dataset_knn, load_dataset

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class KnnResult(object):
    def __init__(self,
        model,
        dataset_name,
        k_neighbors,
        n_points=0,
        ndim=0,
        dir_path=".",
        parameter_name='n_trees',
        experiment_name="exp",
        quality_metric='nnp_rate',
        model_initial_params={},
        model_find_params={},
        save_after_add=False):

        self.model = model
        self.dataset_name = dataset_name
        dataX, dataY = load_dataset(
            dataset_name,
            npoints=n_points,
            ndim=ndim
        )

        self.dataX = dataX
        self.dataY = dataY
        self.n_points = n_points
        self.ndim = ndim
        self.k_neighbors = k_neighbors

        self._dir_path = dir_path
        self._path = os.path.join(dir_path,"{}.json".format(experiment_name))
        self._experiment_name = experiment_name
        self.save_after_add = save_after_add

        self.quality_metric = quality_metric
        self.parameter_name = parameter_name
        self.model_initial_params = model_initial_params
        self.model_find_params = model_find_params

        self.data = {}
        if os.path.exists(self._path):
            with open(self._path) as handle:
                self.data = json.loads(handle.read())
        
        self.parameter_list = []
        self.quality_list = []
        self.time_list = []

    def evaluate(
        self,
        parameter_value,
        model_init_parameter_values=None,
        partition_method=None
    ):
        run_params = self.model_initial_params.copy()
        if model_init_parameter_values is not None:
            run_params.update(model_init_parameter_values)
        model = self.model(**run_params)

        find_params = self.model_find_params.copy()
        find_params[self.parameter_name] = parameter_value

        if partition_method is not None:
            find_params['partition_method'] = partition_method

        t_start = time.time()
        # indices, dist = model.find_nearest_neighbors_ann(
        indices, dist = model.find_nearest_neighbors(
            self.dataX,
            self.k_neighbors,
            **find_params
        )
        t_end = time.time()
        elapsed_time = t_end - t_start

        quality = None
        if self.quality_metric == 'nnp_rate':
            real_distances, real_indices = load_dataset_knn(
                self.dataset_name,
                max_k=self.k_neighbors,
                npoints=self.n_points,
                ndim=self.ndim,
            )
            quality = get_nne_rate(
                real_indices,
                indices,
                random_state=0,
                max_k=self.k_neighbors,
                verbose=0
            )
        else:
            raise ValueError("Unknown quality metric: {}".format(self.quality_metric))
        
        return quality, elapsed_time

    def evaluate_parameter_list(
        self,
        parameter_list,
        verbose=1,
        model_name=None,
        partition_method=None
    ):
        for parameter_value in parameter_list:
            if verbose >=1:
                print("Evaluating parameter {}={}".format(
                    self.parameter_name,
                    parameter_value
                ))
            quality, elapsed_time = self.evaluate(
                parameter_value,
                partition_method=partition_method
            )
            if verbose >=1:
                print("Quality: {:.4f} | Time: {:.4f} sec".format(
                    quality,
                    elapsed_time
                ))
            self.parameter_list.append(parameter_value)
            self.quality_list.append(quality)
            self.time_list.append(elapsed_time)
        

        if model_name is None:
            model_name = self.model.__class__.__name__

        self.add_knn_result(
            self.dataset_name,
            self.k_neighbors,
            model_name,
            self.parameter_name,
            self.parameter_list,
            self.quality_metric,
            self.quality_list,
            self.time_list
        )
        if self.save_after_add:
            self.save()
        return self

    def clean(self):
        self.parameter_list = []
        self.quality_list = []
        self.time_list = []
    
    def add_knn_result(
        self,
        dataset_name,
        K,
        knn_method_name,
        parameter_name,
        parameter_list,
        quality_metric,
        quality_list,
        time_list):
        """
        Add kNN result to the experiment.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset.
        K : int
            Number of neighbors.
        knn_method_name : str
            Name of the kNN method.
        parameter_name : str
            Name of the parameter.
        parameter_list : list
            List of parameter values.
        quality_metric : str
            Name of the quality metric.
        quality_list : list
            List of quality metric values.
        time_list : list
            List of time measurements.
        """
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

        if not quality_metric in obj:
            obj[quality_metric] = {}

        obj[quality_metric] = {
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
    
    def print_summary(self):
        for dataset_name in self.data:
            for K in self.data[dataset_name]:
                for knn_method_name in self.data[dataset_name][K]:
                    for parameter_name in self.data[dataset_name][K][knn_method_name]:
                        data = self.data[dataset_name][K][knn_method_name][parameter_name]
                        if not self.quality_metric in data:
                            continue
                        
                        parameter_list = data[self.quality_metric]["parameters"]
                        quality_list = np.array(data[self.quality_metric]["quality"])
                        time_list = np.array(data[self.quality_metric]["time"])

                        idx = np.argsort(quality_list)

                        print("Dataset: {}, K: {}, Method: {}, Parameter: {}".format(
                            dataset_name,
                            K,
                            knn_method_name,
                            parameter_name
                        ))
                        for i in idx:
                            print("  Param: {:.4f} | Quality: {:.4f} | Time: {:.4f} sec".format(
                                parameter_list[i],
                                quality_list[i],
                                time_list[i]
                            ))
                        print("")
    def plot(
        self,
        dataset_list,
        K,
        quality_metric,
        dataX=None,
        dash_method=[],
        fig_name=None,
        ignore_outliers=True,
        baseline=None,
        method_list=None,
        export_data_to_sheet="plot_data.csv"
    ):
        
        font_default = {
            # 'family' : 'normal',
            # 'weight' : 'bold',
            # 'size'   : 19
            'size'   : 14
        }

        matplotlib.rc('xtick', labelsize=16) 
        matplotlib.rc('ytick', labelsize=16)


        plt.rc('font', **font_default)

        assert (type(dataset_list) == str) or (type(dataset_list) == list)
        if type(dataset_list) is str:
            dataset_list = [dataset_list]


        
        if fig_name is None:
            fig_name = "{}_{}".format(quality_metric,"_".join(dataset_list))+str(self._experiment_name)+"_K{}.pdf".format(K)

        

        fig, ax = plt.subplots(figsize=(10, 6))

        if not baseline is None:
            ivfflat_x = None
            ivfflat_y = None
            # ax2 = ax.twinx()
            plt_colors_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            

        min_x = np.inf
        max_x = 0
        min_y = np.inf
        max_y = 0
        min_y2 = np.inf
        max_y2 = 0
        non_baseline_count = 0

        dataset_legend_pos_rot = ['upper right', 'lower left']
        for dataset_count, dataset_name in enumerate(dataset_list):
            ivfflat_x = None
            ivfflat_y = None

            if dataX is None:
                dataX, dataY = load_dataset(
                    dataset_name,
                    npoints=self.n_points,
                    ndim=self.ndim
                )
            
            dataset_size = len(dataX)

            legend_namelist = []
            legend_curveslist = []
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
                    
                    # legend_name = "({}) ".format(dataset_name) + legend_name
                    
                    if knn_method_name == "IVFFLAT":
                        legend_name = legend_name.replace("IVFFLAT", "FAISS IVFFLAT")
                    if knn_method_name == "IVFPQ":
                        legend_name = legend_name.replace("IVFPQ", "FAISS IVFPQ")
                    if knn_method_name == "FLATL2":
                        legend_name = legend_name.replace("FLATL2", "FAISS FLATL2")
                    
                    if "RSFK-Tiles" in knn_method_name:
                        legend_name = legend_name.replace("RSFK-Tiles", "w-KNNG-Tiles")

                    if "ANNOY" == knn_method_name:
                        legend_name = "ANNOY (CPU)"

                    legend_name = legend_name.replace("GOOGLE_NEWS300_3000000", "GoogleNews300")
                    legend_name = legend_name.replace("GOOGLE_NEWS300", "GoogleNews300")
                    legend_name = legend_name.replace("AMAZON_REVIEW_ELETRONICS", "Amazon Electronics")
                    legend_name = legend_name.replace("LUCID_INCEPTION", "Lucid Inception")
                    legend_name = legend_name.replace("ATSNE_MNIST", "MNIST")
                    legend_name = legend_name.replace("ATSNE_IMAGENET", "Imagenet")
                    legend_name = legend_name.replace("ATSNE_CIFAR", "CIFAR")
                    legend_namelist.append(legend_name)

                    data = self.data[dataset_name][str(K)][knn_method_name][parameter_name]
                    if not quality_metric in data:
                        continue
                    
                    parameter_list = data[quality_metric]["parameters"]
                    quality_list = np.array(data[quality_metric]["quality"])
                    time_list = np.array(data[quality_metric]["time"])

                    

                    
                    time_list = dataset_size/time_list
                    
                    croped_point = None
                    if (not baseline is None) and (not ivfflat_x is None):
                        tmp_idx = np.argsort(quality_list)
                        idx = []
                        for i in tmp_idx:
                            if quality_list[i] >= max(ivfflat_x):
                                break
                            idx.append(i)
                    else:
                        idx = np.argsort(quality_list)

                    curve_x = np.array(quality_list[idx])
                    curve_y = np.array(time_list[idx])

                    if (not baseline is None) and (not ivfflat_x is None) and (max(quality_list) > max(ivfflat_x)):
                        new_point_x = [max(ivfflat_x)]
                        new_point_y = get_projection(curve_x, curve_y, new_point_x)

                        curve_x = np.concatenate((curve_x, np.array(new_point_x)))
                        curve_y = np.concatenate((curve_y, np.array(new_point_y)))
                    
                        # print(max(ivfflat_x), curve_x)
                    if knn_method_name in dash_method:
                        legend_curveslist.append(ax.plot(curve_x, curve_y, ":|", label=legend_name)[0])
                    else:
                        legend_curveslist.append(ax.plot(curve_x, curve_y, "-|", label=legend_name)[0])
                        

                    min_x = min(min_x, min(quality_list[idx]))
                    min_y = min(min_y, min(time_list[idx]))

                    max_x = max(max_x, max(quality_list[idx]))
                    max_y = max(max_y, max(time_list[idx]))
                    
                    if knn_method_name == baseline:
                        ivfflat_x = quality_list[idx]
                        ivfflat_y = time_list[idx]
                        ivfflat_y_total_time = np.array(data[quality_metric]["time"])[idx]
            

            if (not baseline is None) and (not ivfflat_x is None):
                # WORKAROUND
                # ax.set_prop_cycle(cycler('color', plt_colors_cycle[non_baseline_count+1:]))

                curve_count = 0
                for knn_method_name in method_list:
                    for parameter_name in self.data[dataset_name][str(K)][knn_method_name]:
                        if knn_method_name =="IVFFLAT" or knn_method_name in dash_method:
                            continue
                        
                        # legend_name = str(knn_method_name)+" ({})".format(parameter_name)
                        legend_name = str(knn_method_name)
                        
                        if len(dataset_list) > 1:
                            legend_name = "({}) ".format(dataset_name) + legend_name

                        legend_name = legend_name.replace("GOOGLE_NEWS300_3000000", "GoogleNews300")
                        legend_name = legend_name.replace("GOOGLE_NEWS300", "GoogleNews300")
                        legend_name = legend_name.replace("AMAZON_REVIEW_ELETRONICS", "Amazon Electronics")
                        legend_name = legend_name.replace("LUCID_INCEPTION", "Lucid Inception")
                        legend_name = legend_name.replace("ATSNE_MNIST", "MNIST")
                        legend_name = legend_name.replace("ATSNE_IMAGENET", "Imagenet")
                        legend_name = legend_name.replace("ATSNE_CIFAR", "CIFAR")

                        data = self.data[dataset_name][str(K)][knn_method_name][parameter_name]
                        if not quality_metric in data:
                            continue
                        
                        parameter_list = data[quality_metric]["parameters"]
                        quality_list = np.array(data[quality_metric]["quality"])
                        time_list = np.array(data[quality_metric]["time"])

                        
                        idx = np.argsort(quality_list)

                        '''
                        proj = get_projection(ivfflat_x, ivfflat_y, quality_list[idx])
                        fill_time = time_list[idx][proj != None]
                        fill_proj2 = fill_time
                        fill_proj = proj[proj != None]
                        fill_quality = quality_list[idx][proj != None]
                        speedup = fill_time/fill_proj
                        '''

                        curve_x = np.array(quality_list[idx])
                        curve_y = np.array(time_list[idx])

                        if (not baseline is None) and (not ivfflat_x is None) and (max(curve_x) > max(ivfflat_x)):
                            new_point_x = [max(ivfflat_x)]
                            new_point_y = get_projection(curve_x, curve_y, new_point_x)

                            curve_x = np.concatenate((curve_x, np.array(new_point_x)))
                            curve_y = np.concatenate((curve_y, np.array(new_point_y)))

                            
                        # '''
                        fill_quality = np.arange(0.2,min(np.max(curve_x),0.95),0.1)

                        if (max(curve_x) > max(ivfflat_x)):
                            fill_quality = np.concatenate((fill_quality,np.array([max(ivfflat_x)])))
                        
                        proj = get_projection(ivfflat_x, ivfflat_y_total_time, fill_quality)
                        fill_proj = proj[proj != None]
                        
                        proj2 = get_projection(curve_x, curve_y, fill_quality)
                        fill_proj2 = proj2[proj != None]

                        fill_quality = fill_quality[proj != None]

                        # print(fill_proj)
                        # print(fill_proj2)
                        # print("")
                        speedup = fill_proj/fill_proj2
                        # '''

                        fill_proj2 = dataset_size/fill_proj2

                        # print(legend_name, ivfflat_x, quality_list[idx])
                        # print(proj)
                        # print(legend_name, proj, fill_time, fill_proj)
                        # ax2.plot(fill_quality, speedup, ":", label="Speedup " + legend_name)
                        

                        plt.rc('font', size=9)
                        bbox_args = dict(boxstyle='square',
                                         facecolor='white',
                                        #  facecolor=str(plt_colors_cycle[non_baseline_count+curve_count+1]),
                                         edgecolor=str(plt_colors_cycle[non_baseline_count+curve_count+1])
                                         )
                        
                        
                        
                        for i,x in enumerate(fill_quality):
                            if x <= max(ivfflat_x):
                                ax.annotate("{:.2f}".format(speedup[i]), (x, fill_proj2[i]), bbox=bbox_args)

                        # for i in idx:
                        #     if quality_list[i] < max(ivfflat_x):
                        #         ax.annotate("{:.2f}".format(speedup[i]), (quality_list[i], time_list[i]), bbox=bbox_args)
                        

                        plt.rc('font', **font_default)

                        min_y2 = min(min_y2, min(speedup))

                        max_y2 = max(max_y2, max(speedup))

                        curve_count+=1
                        
                    # ax2.ticklabel_format(useOffset=False, style='plain')
                    # ax2.yaxis.get_major_formatter().set_scientific(False)
                    # ax2.legend()
                    # ax2.legend(loc=0)


                # non_baseline_count+=len(method_list)
                non_baseline_count+=len(method_list)+1 #WORKAROUND
                
                # ax2.set_ylabel('Speedup')
                # ax2.set_ylim(1)
                # major_ticks = np.arange(1, max_y2, 2)
                # minor_ticks = np.arange(1, max_y2, 1)
                # ax2.set_yticks(major_ticks)
                # ax2.set_yticks(minor_ticks, minor=True)
                # ax2.grid(which='minor', alpha=0.2)
                # ax2.grid(which='major', alpha=0.5)

            # fig.legend()

            # legend_curveslist.append(ax.plot([], [], label=" ")[0])
            # legend_curveslist.append(ax.plot([], [], label="Speedups over FAISS")[0])
            first_legend = plt.legend(handles=legend_curveslist, loc=dataset_legend_pos_rot[dataset_count], bbox_transform=ax.transAxes, framealpha=0.5, prop={'size': 11})
            # Add the legend manually to the current Axes.
            plt.gca().add_artist(first_legend)

            # ax = plt.gca().add_artist(first_legend)
            # WORKAROUND
            



        # ax.legend()
        # fig.legend(loc="lower left", bbox_to_anchor=(0,0), bbox_transform=ax.transAxes, framealpha=0.5, prop={'size': 14})
        # fig.legend()
        # plt.legend()

        '''
        # major_ticks = np.arange(min_x, max_x, 1)
        # minor_ticks = np.arange(min_x, max_x, 0.5)
        # ax.set_yticks(major_ticks)
        # ax.set_yticks(minor_ticks, minor=True)
        '''

        # major_ticks = np.arange(1, max_y2, 1)
        # minor_ticks = np.arange(0, max_y2, 0.5)
        major_ticks = np.arange(0.1,0.95,0.1)
        # major_ticks = np.concatenate((major_ticks,np.array([max(ivfflat_x)])))
        minor_ticks = np.arange(0.1,0.95,0.05)
        # minor_ticks = np.concatenate((minor_ticks,np.array([max(ivfflat_x)])))
        
        ax.set_xticks(major_ticks)
        # ax.set_xticks(minor_ticks, minor=True)


        # for label in ax.xaxis.get_ticklabels():
        #     label.set_bbox(dict(facecolor='none', edgecolor='black'))

        
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)
        # ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))

        # And a corresponding grid
        ax.grid(which='both')

        # Or if you want different settings for the grids:
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)



        # ax.set_xlabel('K-NNG Accuracy')
        # ax.set_ylabel('Average of points treated per second')
        ax.set_xlabel('Acurácia')
        ax.set_ylabel('Média de pontos processados por segundo')
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
            if quality_metric == "dist_mean":
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
        
        fig.savefig(fig_name)

    def export_time_accuracy_to_csv(self, dataset_list, K, quality_metric, dataX=None, 
             dash_method=[], fig_name=None, ignore_outliers=True, baseline=None, method_list=None, sheet_name="plot_data.xls"):
        

        assert (type(dataset_list) == str) or (type(dataset_list) == list)
        if type(dataset_list) is str:
            dataset_list = [dataset_list]


        

        fig, ax = plt.subplots(figsize=(10, 6))

        if not baseline is None:
            ivfflat_x = None
            ivfflat_y = None
            # ax2 = ax.twinx()
            plt_colors_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            

        min_x = np.inf
        max_x = 0
        min_y = np.inf
        max_y = 0
        min_y2 = np.inf
        max_y2 = 0
        non_baseline_count = 0

        column_list_name = []
        column_list_data = []

        for dataset_count, dataset_name in enumerate(dataset_list):
            ivfflat_x = None
            ivfflat_y = None

            dataX, dataY = load_dataset(
                dataset_name,
                npoints=self.n_points,
                ndim=self.ndim
            )
            dataset_size = len(dataX)

            legend_namelist = []
            legend_curveslist = []
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
                    
                    legend_name = str(knn_method_name)
                    if len(dataset_list) > 1:
                        legend_name = "({}) ".format(dataset_name) + legend_name
                    
                    if knn_method_name == "IVFFLAT":
                        legend_name = legend_name.replace("IVFFLAT", "FAISS IVFFLAT")
                    if knn_method_name == "IVFPQ":
                        legend_name = legend_name.replace("IVFPQ", "FAISS IVFPQ")
                    if knn_method_name == "FLATL2":
                        legend_name = legend_name.replace("FLATL2", "FAISS FLATL2")
                        
                    legend_name = legend_name.replace("GOOGLE_NEWS300", "GoogleNews300")
                    legend_name = legend_name.replace("AMAZON_REVIEW_ELETRONICS", "Amazon Electronics")
                    legend_name = legend_name.replace("LUCID_INCEPTION", "Lucid Inception")
                    legend_name = legend_name.replace("ATSNE_MNIST", "MNIST")
                    legend_name = legend_name.replace("ATSNE_IMAGENET", "Imagenet")
                    legend_name = legend_name.replace("ATSNE_CIFAR", "CIFAR")
                    legend_namelist.append(legend_name)

                    data = self.data[dataset_name][str(K)][knn_method_name][parameter_name]
                    if not quality_metric in data:
                        continue
                    
                    parameter_list = data[quality_metric]["parameters"]
                    quality_list = np.array(data[quality_metric]["quality"])
                    time_list = np.array(data[quality_metric]["time"])

                    column_list_name.append("{} - Time".format(legend_name))
                    column_list_data.append(time_list)

                    column_list_name.append("{} - Accuracy".format(legend_name))
                    column_list_data.append(quality_list)


                    time_list = dataset_size/time_list

                    column_list_name.append("{} - Points/second".format(legend_name))
                    column_list_data.append(time_list)
                    
                    # croped_point = None
                    # if (not baseline is None) and (not ivfflat_x is None):
                    #     tmp_idx = np.argsort(quality_list)
                    #     idx = []
                    #     for i in tmp_idx:
                    #         if quality_list[i] >= max(ivfflat_x):
                    #             break
                    #         idx.append(i)
                    # else:
                    #     idx = np.argsort(quality_list)

                    # curve_x = np.array(quality_list[idx])
                    # curve_y = np.array(time_list[idx])

                    
                    
                    # if (not baseline is None) and (not ivfflat_x is None):
                    #     new_point_x = [max(ivfflat_x)]
                    #     new_point_y = get_projection(curve_x, curve_y, new_point_x)

                    #     curve_x = np.concatenate((curve_x, np.array(new_point_x)))
                    #     curve_y = np.concatenate((curve_y, np.array(new_point_y)))
                    
                    #     # print(max(ivfflat_x), curve_x)
                    # if knn_method_name in dash_method:
                    #     legend_curveslist.append(ax.plot(curve_x, curve_y, ":|", label=legend_name)[0])
                    # else:
                    #     legend_curveslist.append(ax.plot(curve_x, curve_y, "-|", label=legend_name)[0])
                        

                    # min_x = min(min_x, min(quality_list[idx]))
                    # min_y = min(min_y, min(time_list[idx]))

                    # max_x = max(max_x, max(quality_list[idx]))
                    # max_y = max(max_y, max(time_list[idx]))
                    
                    # if knn_method_name == baseline:
                    #     ivfflat_x = quality_list[idx]
                    #     ivfflat_y = time_list[idx]
                    #     ivfflat_y_total_time = np.array(data[quality_metric]["time"])[idx]
            
            


        df_list = []
        
        for cd, cn in zip(column_list_data,column_list_name):
            df = pd.DataFrame({cn:cd})
            df_list.append(df)

        df = pd.concat(df_list, ignore_index=True, axis=1)
        df.columns = column_list_name
        # df.to_csv(sheet_name, columns=column_list_name, header=True, index=False)
        df.to_excel(sheet_name, header=True, index=False)

        # df = pd.DataFrame(column_list_data, columns=column_list_name)
        # df.to_csv(sheet_name, columns=column_list_name, header=True, index=False)
        