"""
Module implementation of beyond neural scaling laws beating power scaling laws through data pruning
"""
import argparse
import os
import shutil
import torch
from skfuzzy import cmeans_predict
from sklearn.cluster import KMeans
import skfuzzy as fuzz
import prosemble as  ps
import pandas as pd
import numpy as np
from tqdm import tqdm
from .gng import GrowingNeuralGas


def get_extracted_audio_features(file_list: list):
    """

    :param file_list: list: list of audio files  
    :return: array-like: extracted features and corresponding audio file names
    """
    all_features = []
    for index, file in enumerate(file_list):
        feature_file = file.replace('.wav', '.pt')
        features = torch.load(feature_file).detach().cpu().numpy().flatten()
        all_features.append(features)
    return all_features, file_list

def get_cluster_distance_space(distance_space, file_names):
    """


    :param distance_space: array-like: 2D list containing the distance space
        wrt to the centroids
    :param file_names: list: images names
    :return: clustered distance space for a given cluster label.
    """

    subspace_information_list, subspace_list, counter = [], [], []
    label_basis, sorted_subspace = [], []
    nearest_sort, cluster_distance_space = [], []

    distance_space_information = [
        [file_names[sub_space_index], np.argmin(distance_subspace),
         np.min(distance_subspace)]
        for sub_space_index, distance_subspace in enumerate(distance_space)
    ]

    number_cluster = len(np.unique(
        [i[1] for i in distance_space_information])
    )

    for label in range(number_cluster):
        count = 0
        for index, distance_subspace_information in \
                enumerate(distance_space_information):
            if distance_subspace_information[1] == label:
                count += 1
                subspace_list.append(
                    distance_subspace_information
                )
        counter.append(count)

    # get data subspace information on label basis
    get_sorted_space(
        counter,
        subspace_information_list,
        subspace_list
    )

    for subspace_information in subspace_information_list:
        for index, subspace in enumerate(subspace_information):
            label_basis.append(subspace[2])

    #  get the distance subspace information based on labels
    get_sorted_space(
        counter,
        sorted_subspace,
        label_basis
    )

    # get sorted subspace index based on closest prototype
    for i in sorted_subspace:
        nearest_sort.append(np.argsort(i))

    # get the sort files base on subspace information
    for i in range(len(nearest_sort)):
        cluster_distance_space.append(
            [subspace_information_list[i][v][0]
             for j, v in enumerate(nearest_sort[i])]
        )

    return cluster_distance_space


def get_sorted_space(x, y, z):
    init_count = 0
    for count in x:
        count = count + init_count
        y.append(z[init_count:count])
        init_count = count


def get_pruned_easy_hard_examples(sorted_distance_space, prune_fraction):
    """

    :param sorted_distance_space: list: list with indexes
        of sorted distanced space
    :param prune_fraction: float: prune percentage or fraction
    :return: array-like: pruned list with indices of sorted distance space
    """

    maximum_prune = int(
        len(sorted_distance_space) -
        np.ceil((1 - prune_fraction) * len(sorted_distance_space))
    )
    return sorted_distance_space[:maximum_prune], \
        sorted_distance_space[maximum_prune:]


def get_prune_set(pruned_indexes, prune_mode):
    """

    :param pruned_indexes:list: list with pruned indexes
    :param prune_mode:str: easy , hard and both. If none default is both
    :return: array-like: list with file names of pruned data set.
    """
    if prune_mode == 'easy':
        return pruned_indexes[0]
    if prune_mode == 'hard':
        return pruned_indexes[1]
    if prune_mode == 'both':
        return pruned_indexes
    return None


def get_pruned(distance_space, file_names, prune_fraction, prune_labels):
    """

    :param prune_labels:
    :param prune_fraction:float: fraction of the prune
    :param distance_space:computed distances from the nearest prototype
    :param file_names:
    :return:
    """
    clustered_distance_space = get_cluster_distance_space(
        distance_space,
        file_names
    )
    pruned_all = []
    if len(prune_labels) > 1:
        labels = 'both'
    else:
        labels = prune_labels[0]
    for i in range(len(clustered_distance_space)):
        pruned = get_prune_set(get_pruned_easy_hard_examples(
            clustered_distance_space[i],
            prune_fraction),
            labels
        )
        pruned_all.append(pruned)
    return pruned_all
    

def check_directory(number_clusters, folder_name):
    """

    :param number_clusters: int: number of clusters
    :param folder_name: str: folder name
    :return: cleans the directory for new runs regarding new runs
    """
    directory = [
        f"./{folder_name}_{folder_index}"
        for folder_index in range(number_clusters)
    ]
    for folder in directory:
        if os.path.exists(folder) and os.path.isdir(folder):
            shutil.rmtree(folder, ignore_errors=True)
        if not os.path.exists(folder):
            os.makedirs(folder)


def check_directory_prune(number_clusters, folder_name):
    """

    :param number_clusters: int: number of clusters
    :param folder_name: str: folder name
    :return: cleans the directory for new runs regarding new runs
    """

    directory = [
        [f"./{folder_name}_{cluster_label}_{prune_type}"
         for prune_type in ['easy', 'hard']] for cluster_label
        in range(number_clusters)
    ]

    directory_prune = [
        folder for prune_folders in directory for folder in prune_folders
    ]

    for folder in directory_prune:
        if os.path.exists(folder) and os.path.isdir(folder):
            shutil.rmtree(folder, ignore_errors=True)
        if not os.path.exists(folder):
            os.makedirs(folder)


def ssl_growing_neural_gas(x, y=2):
    gng = GrowingNeuralGas(np.array(x).transpose())
    gng.fit_network(
        e_b=0.1,
        e_n=0.006,
        a_max=10,
        l=200,
        a=0.5,
        d=0.995,
        passes=y,
        plot_evolution=False
    )
    return gng.number_of_clusters()


class SSL:
    def __init__(self, number_cluster, random_state, dataframe):
        self.number_cluster = number_cluster
        self.random_state = random_state
        self.dataframe = dataframe
        self.file_list = dataframe['fn'].values.tolist()
        self.number_topology = []

    def get_embedded_space(self):
        image_features, file_names = get_extracted_audio_features(
            file_list=self.file_list
        )
        return image_features, file_names

    def ssl_kmeans(self, init='k-means++', n_init='auto'):
        embedded_space, file_names = self.get_embedded_space()
        number_of_clusters = self.get_number_clusters(embedded_space)
        self.number_topology.append(number_of_clusters)
        self_supervised_learning_model = KMeans(
            n_clusters=number_of_clusters,
            random_state=self.random_state,
            init=init,
            n_init=n_init
        )
        self_supervised_learning_model.fit(embedded_space)

        prototype_responsibilities = self_supervised_learning_model.fit_transform(
            embedded_space
        )

        cluster_labels = self_supervised_learning_model.labels_
        cluster_centers = self_supervised_learning_model.cluster_centers_

        return prototype_responsibilities, cluster_labels, \
            file_names, cluster_centers, embedded_space

    def ssl_fcm(self):
        embedded_space, file_names = self.get_embedded_space()
        number_of_clusters = self.get_number_clusters(embedded_space)
        self.number_topology.append(number_of_clusters)
        cntr, u_matrix, u_matrix_init, distance_space, \
            objective_function_history, num_inter_run, \
            fuzzy_partition_coefficient = fuzz.cmeans(
                data=np.array(embedded_space).transpose(),
                c=number_of_clusters,
                m=2,
                error=0.001,
                maxiter=1000,
                init=None,
                seed=self.random_state
            )
        cluster_labels = [np.argmax(i) for i in u_matrix.transpose()]

        return distance_space.transpose(), cluster_labels, file_names

    def ssl_fcm_init(self):
        prototype_responsibilities, cluster_labels, \
            file_names, cluster_centers, embedded_space = self.ssl_kmeans()
        u_matrix, u_matrix_init, distance_space, object_function_history, \
            num_iter_run, fuzzy_partition_coefficient = cmeans_predict(
                test_data=np.array(embedded_space).transpose(),
                cntr_trained=cluster_centers,
                m=2,
                error=0.001,
                maxiter=1000,
                init=None,
                seed=self.random_state
        )

        cluster_labels = [np.argmax(i) for i in u_matrix.transpose()]
        return distance_space.transpose(), cluster_labels, file_names

    def ssl_pcm(self):
        embedded_space, file_names = self.get_embedded_space()
        number_of_clusters = self.get_number_clusters(embedded_space)
        self.number_topology.append(number_of_clusters)
        self_supervised_learning_model = ps.models.pcm.PCM(
            data= embedded_space,
            c=number_of_clusters,
            epsilon=0.001,
            num_iter=1000,
            m=2,
            k=1,
            ord='fro',
            set_U_matrix='fcm',
            plot_steps=False
        )
        self_supervised_learning_model.fit()
        cluster_labels = self_supervised_learning_model.predict()
        distance_space = self_supervised_learning_model.get_distance_space(
            embedded_space
        )
        return distance_space,cluster_labels,file_names

    def ssl_fpcm(self):
        embedded_space, file_names = self.get_embedded_space()
        number_of_clusters = self.get_number_clusters(embedded_space)
        self.number_topology.append(number_of_clusters)
        self_supervised_learning_model = ps.models.fpcm.FPCM(
            data=embedded_space,
            c=number_of_clusters,
            epsilon=0.001,
            num_iter=1000,
            m=2,
            eta=2,
            ord=None,
            set_U_matrix='fcm',
            plot_steps=False
        )
        self_supervised_learning_model.fit()
        cluster_labels = self_supervised_learning_model.predict()
        distance_space = self_supervised_learning_model.get_distance_space(
            embedded_space
        )
        return distance_space, cluster_labels, file_names

    def ssl_pfcm(self):
        embedded_space, file_names = self.get_embedded_space()
        number_of_clusters = self.get_number_clusters(embedded_space)
        self.number_topology.append(number_of_clusters)
        self_supervised_learning_model = ps.models.pfcm.PFCM(
            data=embedded_space,
            c=number_of_clusters,
            epsilon=0.001,
            num_iter=1000,
            m=2,
            k=1,
            eta=2,
            a=2,
            b=2,
            ord=None,
            set_U_matrix='fcm',
            plot_steps=False
        )
        self_supervised_learning_model.fit()
        cluster_labels = self_supervised_learning_model.predict()
        distance_space = self_supervised_learning_model.get_distance_space(
            embedded_space
        )
        return distance_space, cluster_labels, file_names

    def ssl_ipcm(self):
        embedded_space, file_names = self.get_embedded_space()
        number_of_clusters = self.get_number_clusters(embedded_space)
        self.number_topology.append(number_of_clusters)
        self_supervised_learning_model = ps.models.ipcm.IPCM1(
            data=embedded_space,
            c=number_of_clusters,
            epsilon=0.001,
            num_iter=None,
            m_f=2,
            m_p=2,
            k=2,
            ord=None,
            set_centroids=None,
            set_U_matrix='fcm',
            plot_steps=False
        )
        self_supervised_learning_model.fit()
        cluster_labels = self_supervised_learning_model.predict()
        distance_space = self_supervised_learning_model.get_distance_space(
            embedded_space
        )
        return distance_space, cluster_labels, file_names

    def ssl_ipcm_2(self):
        embedded_space, file_names = self.get_embedded_space()
        number_of_clusters = self.get_number_clusters(embedded_space)
        self.number_topology.append(number_of_clusters)
        self_supervised_learning_model = ps.models.ipcm_2.IPCM2(
            data=embedded_space,
            c=number_of_clusters,
            epsilon=0.001,
            num_iter=1000,
            m_f=2,
            m_p=2,
            ord=None,
            set_U_matrix='fcm',
            plot_steps=False
        )
        self_supervised_learning_model.fit()
        cluster_labels = self_supervised_learning_model.predict()
        distance_space = \
            self_supervised_learning_model.get_distance_space(embedded_space)
        return distance_space, cluster_labels, file_names

    def ssl_bgpc(self):
        embedded_space, file_names = self.get_embedded_space()
        number_of_clusters = self.get_number_clusters(embedded_space)
        self.number_topology.append(number_of_clusters)
        self_supervised_learning_model = ps.models.bgpc.BGPC(
            data=embedded_space,
            c=number_of_clusters,
            epsilon=0.001,
            num_iter=100,
            a_f=0.8,
            b_f=0.004,
            ord=None,
            set_centroids=None,
            set_U_matrix=None,
            plot_steps=False
        )
        self_supervised_learning_model.fit()
        cluster_labels = self_supervised_learning_model.predict()
        distance_space = self_supervised_learning_model.get_distance_space(
            embedded_space
        )
        return distance_space, cluster_labels, file_names

    def ssl_hcm(self):
        embedded_space, file_names = self.get_embedded_space()
        number_of_clusters = self.get_number_clusters(embedded_space)
        self.number_topology.append(number_of_clusters)
        self_supervised_learning_model = ps.models.hcm.Kmeans(
            data=np.array(embedded_space),
            c=number_of_clusters,
            epsilon=0.001,
            num_inter=1000,
            ord=None,
            set_prototypes=None,
            plot_steps=False
        )
        self_supervised_learning_model.fit()
        cluster_labels = self_supervised_learning_model.predict()
        distance_space = self_supervised_learning_model.get_distance_space(
            embedded_space
        )
        return distance_space, cluster_labels, file_names

    def get_number_clusters(self, x):
        """

        :param x: embeded space
        :return: number of clusters.
        """
        if isinstance(self.number_cluster, int):
            return self.number_cluster
        if self.number_cluster == 'default':
            number_cluster = 2
            return number_cluster
        if self.number_cluster == 'auto':
            learned_topologies = ssl_growing_neural_gas(x)
            if learned_topologies < 2:
                number_cluster = 2
                return number_cluster
            return learned_topologies
        return None


class Prune(SSL):
    """
    Prune
    params:

    random_state: int:
        Random seed
    prune_fraction: float:
        fraction or percentage of dataset to prune
    prune_mode: str:
        easy , hard and both
    prune_type: bool:
        True or False .Indicating the ceiling or floor of the prune results.


    """

    def __init__(self, prune_type, ssl_type, data_frame_clustered,
                 number_cluster, random_state, dataframe):
        super().__init__(number_cluster, random_state, dataframe)
        self.prune_mode = ['easy', 'hard']
        # self.prune_mode = ['hard']
        self.prune_type = prune_type
        self.prune_folder_name = 'prune'
        self.ssl_type = ssl_type
        self.date_frame_clustered = data_frame_clustered

        if self.ssl_type == 'kmeans':
            self.distance_space, self.clustered_labels, \
                self.clustered_file_names, self.cluster_centers, \
                self.embedded_space = self.ssl_kmeans()

        if self.ssl_type == "fcm":
            self.distance_space, self.clustered_labels, \
                self.clustered_file_names = self.ssl_fcm()

        if self.ssl_type == "fcm_init":
            self.distance_space, self.clustered_labels, \
                self.clustered_file_names = self.ssl_fcm_init()

        if self.ssl_type == "pcm":
            self.distance_space, self.clustered_labels, \
                self.clustered_file_names = self.ssl_pcm()

        if self.ssl_type == "fpcm":
            self.distance_space, self.clustered_labels, \
                self.clustered_file_names = self.ssl_fpcm()

        if self.ssl_type == "pfcm":
            self.distance_space, self.clustered_labels, \
                self.clustered_file_names = self.ssl_pfcm()

        if self.ssl_type == "ipcm":
            self.distance_space, self.clustered_labels, \
                self.clustered_file_names = self.ssl_ipcm()

        if self.ssl_type == "ipcm_2":
            self.distance_space, self.clustered_labels, \
                self.clustered_file_names = self.ssl_ipcm_2()

        if self.ssl_type == "bgpc":
            self.distance_space, self.clustered_labels, \
                self.clustered_file_names = self.ssl_bgpc()

        if self.ssl_type == "hcm":
            self.distance_space, self.clustered_labels, \
                self.clustered_file_names = self.ssl_hcm()
        
        cluster_df = pd.DataFrame({'fn':self.clustered_file_names, 'cluster':self.clustered_labels, 'distance':self.distance_space.tolist()})
        self.dataframe = pd.merge(self.dataframe, cluster_df, on='fn')

    def get_number_topologies(self):
        """

        :return: The learned topologies.
        """
        return self.number_topology

    def get_cluster_results(self):
        """

        :return: panda dataframe and populate folders wit the clustering results.
        """
        if isinstance(self.number_cluster, str):
            self.number_cluster = self.get_number_topologies()[0]

        image_cluster_df = pd.DataFrame(
            self.clustered_file_names,
            columns=['file_names']
        )

        image_cluster_df["cluster_label"] = self.clustered_labels

        return image_cluster_df


    def prune(self, prune_fraction):

        if isinstance(self.number_cluster, str):
            self.number_cluster = self.get_number_topologies()[0]
            

        all_pruned=get_pruned(
            distance_space=self.distance_space,
            file_names=self.clustered_file_names,
            prune_fraction=prune_fraction,
            prune_labels=self.prune_mode
        )
        selected = []
        for pruned in all_pruned:
            selected.extend(pruned[0])
            
        return self.dataframe.loc[self.dataframe['fn'].isin(selected)]
        
