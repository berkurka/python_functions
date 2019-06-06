from sklearn.cluster import KMeans,AgglomerativeClustering
from scipy.cluster import hierarchy
from sklearn.metrics import silhouette_score
import pandas as pd

def grid_clusters(df_dic, par_dic):
    result_dic = {"Data_frame" :[],
                  "model": [],
                  "inertia": [],
                  "silhouette": [],
                  "Numb_clusters": [],
                  "Cluster_counts": [],
                  "model_params": []
                  }
    # loop dataFrames
    for df_name, df_data in df_dic.items():
        X = df_data

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
                                #Saving results in dic
                                result_dic['Data_frame'].append(df_name)
                                result_dic['model'].append(n_model)
                                result_dic['inertia'].append(kmeans.inertia_)
                                result_dic['silhouette'].append(silhouette_score(X, kmeans.labels_))
                                result_dic['Numb_clusters'].append(n_clusters)
                                result_dic['Cluster_counts'].append(list(pd.Series(kmeans.labels_).value_counts()))
                                result_dic['model_params'].append(par_dic['kmeans'])

            if n_model == "hierarchy":
                # loop linkage_method
                for n_link_met in par_dic['hierarchy']['linkage_method']:
                    for n_clusters in par_dic['hierarchy']['t']:
                        for n_crit in par_dic['hierarchy']['criterion']:
                            z = hierarchy.linkage(X, method=n_link_met)
                            cls = hierarchy.fcluster(z, n_clusters, n_crit)
                            # Saving results in dic
                            result_dic['Data_frame'].append(df_name)
                            result_dic['model'].append(n_model)
                            result_dic['inertia'].append(0)
                            result_dic['silhouette'].append(silhouette_score(X, cls))
                            result_dic['Numb_clusters'].append(n_clusters)
                            result_dic['Cluster_counts'].append(list(pd.Series(cls).value_counts()))
                            result_dic['model_params'].append(par_dic['hierarchy'])
            if n_model == "Agglomerative":
                # loop linkage criterion
                for n_link_met in par_dic['Agglomerative']['linkage_method']:
                    #loop number of clusters
                    for n_clusters in par_dic['Agglomerative']['clusters_list']:
                        # loop method to compute linkage
                        for n_affinity in par_dic['Agglomerative']['affinity']:
                            ac = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
                            ac.fit(X)
                            # Saving results in dic
                            result_dic['Data_frame'].append(df_name)
                            result_dic['model'].append(n_model)
                            result_dic['inertia'].append(0)
                            result_dic['silhouette'].append(silhouette_score(X, ac.labels_))
                            result_dic['Numb_clusters'].append(n_cluster)
                            result_dic['Cluster_counts'].append(list(pd.Series(ac.labels_).value_counts()))
                            result_dic['model_params'].append(par_dic['Agglomerative'])
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

# EXAMPLE OF MODEL DIC
# dic = {"kmeans": {"init_seed": [10],
#                   "inits"  :  ["k-means++","random"],
#                   "algorithms": ["auto", "full", "elkan"],
#                   "clusters_list": range(4,10),
#                   "tol": 0.0001},
#        "hierarchy": {"linkage_method":['complete','ward', 'single',
#                                        'centroid','median','weighted'],
#                      "t": range(4,10),
#                      "criterion": "maxclust"},
#        "Agglomerative": {"linkage_method":['complete','ward', 'single',
#                                            'centroid','median','weighted'],
#                          "affinity": ['euclidean', 'l1', 'l2', 'manhattan',
#                                       'cosine', 'precomputed'],
#                          "clusters_list": range(4,10)}
       }
# EXAMPLE OF DF DIC
# df_dic = {'All_features': data[all_features], 'Raw_prices': data[raw_prices],
#          'scaled_prices': data[sc_prices],'scaled_pr_mult': data[sc_multip_prices],
#          'scaled_pr_mult_wtot': data[sc_multip_prices_w_tot]
#          }

