import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from numpy.random import choice, seed
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

seed(42)


def initialize_medoids(num_medoids, data):
    return [data.iloc[idx] for idx in choice(len(data), size=num_medoids, replace=False)]


def assign_points_to_medoids(data, medoids):
    return [np.argmin([distance_vec2vec(point[1], medoid) for medoid in medoids]) for point in data.iterrows()]


def distance_vec2vec(a, b) -> np.float64:
    return sum([(abs(a[i] - b[i]) ** 2) for i in range(len(a))])


def reassign_medoids(data, assignments, initial_medoids):
    new_medoids = []
    for idm, medoid in enumerate(initial_medoids):
        new_medoid = medoid
        medoid_score = sum([distance_vec2vec(medoid, x[1]) if assignments[idx] == idm else 0
                            for idx, x in enumerate(data.iterrows())])
        for point in data.iterrows():
            point_score = sum(sum([distance_vec2vec(point, x[1]) if assignments[idx] == idm else 0
                                   for idx, x in enumerate(data.iterrows())]))
            if medoid_score > point_score:
                new_medoid = point
        new_medoids.append(new_medoid)
    return new_medoids


def is_finished(old_medoids, new_medoids):
    return set([tuple(om) for om in old_medoids]) == set([tuple(nm) for nm in new_medoids])


def kmedoids(num_samples, num_clusters):
    df = pd.read_csv('CC GENERAL.csv', index_col='CUST_ID')
    df = df[:num_samples].fillna(0)
    df_scaled = pd.DataFrame(StandardScaler().fit_transform(df))

    # initialize medoids (at random)
    medoids = initialize_medoids(num_medoids=num_clusters, data=df_scaled)

    # assign data points to the medoids
    assignments = assign_points_to_medoids(data=df_scaled, medoids=medoids)

    # fit
    new_medoids = reassign_medoids(data=df_scaled, assignments=assignments, initial_medoids=medoids)
    while not is_finished(old_medoids=medoids, new_medoids=new_medoids):
        medoids = new_medoids
        new_medoids = reassign_medoids(data=df_scaled, assignments=assignments, initial_medoids=medoids)

    new_assignments = assign_points_to_medoids(data=df_scaled, medoids=new_medoids)
    data = pd.DataFrame(PCA(n_components=2).fit_transform(df_scaled), columns=['0', '1'])
    data['cluster'] = new_assignments

    sns.relplot(x='0', y='1', hue='cluster', data=data, palette=sns.color_palette("husl", num_clusters))
    plt.show()


for i in range(2, 8):
    kmedoids(num_samples=500, num_clusters=i)
    print(i)
