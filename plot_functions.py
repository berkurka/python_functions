import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.cluster import hierarchy
from sklearn.cluster import KMeans,DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram

def single_histogram(list,y_label,x_label):
    """
    :param list: Numeric List;
        Data to create histogram.
    :param y_lable: str;
        Name of Y lable.
    :param x_lable: str;
        Name of X lable.
    :return: none
        Prints a plot, no return
    Example:
    ages = [0,10,21,31,14,15,10,20]
    single_histogram(ages,"Age Frequency","Age")
    """

    # Mean, std
    mu = str(round(np.mean(list),2))
    sigma = str(round(np.std(list)))
    mean= str(round(np.mean(list)))
    # the histogram of the data
    n, bins, patches = plt.hist(list, facecolor='b', alpha=0.75)
    print('mean=' + mean)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title('Histogram of ' + x_label)

    #Printing mean and std
    plt.text(max(list)/2,max(list)/1.8,
             r'$\mu='+ mu +',\ \sigma=' + sigma + '$'
             ,horizontalalignment='center'
             ,verticalalignment='top'
             ,fontsize=10)
    #Ajust axis here: [X0,X1,Y0,Y1]
    # plt.axis([5, 10, 0, 20])
    plt.grid(True)
    plt.show()

def hierarchy_plot(X,p_input,truncate,link_method_list):
    """
    :param X: Pandas DF
        Data to cluster.
    :param p_input: int
        The ``p`` parameter for ``truncate_mode``.
    :param truncate: str, optional
        Truncation is used to condense the dendrogram.
        Example: ``None``,``'lastp'``, ``'level'``.
    :param link_method_list: list of str
        Methods used to compute the distance.
        Example: 'ward', 'single', 'centroid'...
    :return: nothing,
        Plots dendograms

    Example:
        hierarchy_plot(X,30,'lastp',['complete','ward', 'single', 'centroid'])
    """
    # Generating Linkage list
    z_list = [hierarchy.linkage(X, method=z) for z in link_method_list]
    # Color Pallete set up
    hierarchy.set_link_color_palette(['m', 'c', 'y', 'k'])

    # Setting Sub Plots
    fig, axes = plt.subplots(max(2,len(link_method_list)//2),
                             2,
                             figsize=(20, 25))
    #
    for pos,Z in enumerate(z_list):
        pos_x = min(pos // 2 + pos % 2, pos // 2)
        pos_y = max(pos % 2, pos % 2)
        hierarchy.dendrogram(Z, ax=axes[pos_x][pos_y], p=p_input, above_threshold_color='y',
                             orientation='right', truncate_mode=truncate)
        axes[pos_x, pos_y].set_title("Metric: " + link_method_list[pos]
                                     + ", p=" +str(p_input)
                                     + ", truncate = " + truncate
                                     , fontsize=15)

    hierarchy.set_link_color_palette(None)
    plt.figure()


def grid_clusters(X, par_dic):
    result_dic = {"model": [],
                  "inertia": [],
                  "silhouette": [],
                  "model_params": []
                  }
    # loop models
    for n_model in par_dic.keys():

        if n_model == "kmeans":
            # loop centroid seeds
            for n_init_seed in par_dic['kmeans']['init_seed']:
                # loop Method for initialization
                for n_init_method in par_dic['kmeans']['inits']:
                    # loop algorithms
                    for n_algorithm in par_dic['kmeans']['algorithms']:
                        # loop number of clusters
                        for n_cluster in par_dic['kmeans']['clusters_list']:
                            kmeans = KMeans(n_clusters=n_cluster,
                                            init=n_init_method,
                                            n_init=n_init_seed,
                                            algorithm=n_algorithm,
                                            tol=par_dic['kmeans']['tol'],
                                            random_state=42)
                            kmeans.fit(X)
                            result_dic['model'].append(n_model)
                            result_dic['inertia'].append(kmeans.inertia_)
                            result_dic['silhouette'].append(silhouette_score(X, kmeans.labels_))
                            result_dic['model_params'].append(par_dic['kmeans'])

    #         if n_model == "DBSCAN":
    #             #Loop maximum distance between two samples
    #             for n_eps in par_dic['DBSCAN']['eps']:
    #                 #loop min number of samples
    #                 for n_min_samples in par_dic['DBSCAN']['min_samples']:
    #                     #loop distance metric
    #                     for n_metric in par_dic['DBSCAN']['metric']:
    #                         #loop algorithm used by the NearestNeighbors module to compute pointwise
    #                         for n_algorithm in par_dic['DBSCAN']['algorithm']:
    #                             dbsc = DBSCAN(eps =n_eps,
    #                                           min_samples = n_min_samples,
    #                                           metric=n_metric,
    #                                           algorithm=n_algorithm,
    #                                           leaf_size=30
    #                                          )
    #                             dbsc.fit(X)
    #                             result_dic['model'].append(n_model)
    #                             result_dic['inertia'].append(dbsc.inertia_)
    #                             result_dic['silhouette'].append(silhouette_score(X, dbsc.labels_))
    #                             result_dic['model_params'].append(dic['DBSCAN'])

    return pd.DataFrame(result_dic)

