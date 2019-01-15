import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram

def single_histogram(list,y_label,x_label):
    """
    :param list: Numeric List;
    Data to create histogram.
    :param y_lable: String;
    Name of Y lable.
    :param x_lable: String;
    Name of X lable.
    :return:
    Prints a plot, no return
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

    #Printing
    plt.text(max(list)/2,max(list)/1.1,
             r'$\mu='+ mu +',\ \sigma=' + sigma + '$'
             ,horizontalalignment='center'
             ,verticalalignment='center'
             ,fontsize=20)
#   plt.axis([40, 160, 0, 0.03])
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
