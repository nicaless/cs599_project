
# coding: utf-8

# In[155]:

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy
import scipy.cluster.hierarchy as hcluster
import pandas as pd
import glob
import os
import numpy as np
import cv2
import sys

vid_name = sys.argv[1]


# In[182]:

# Velocity Trace - Fast Cars

for f in glob.glob('projections/' + vid_name + '_*.csv'):
    # Get Video ID
    vid_id =  f.split("projections/")[1].split("_")[0]
    # Read in Data
    data = pd.read_csv(f, header=None)
    if "fast_cars" in f:
        data.columns = ["Vid", "Object", "C"]
        cluster_var = ['C']
        save_name = "_fast_cars"
        data_group_name = "_withVelocity.csv"
        data_cols = ['cog_x','cog_y','frame','height','obj','velocity','width','x','y']
    elif "slow_cars" in f:
        data.columns = ["Vid", "Object", "C"]
        cluster_var = ['C']
        save_name = "_slow_cars"
        data_group_name = "_withVelocity.csv"
        data_cols = ['cog_x','cog_y','frame','height','obj','velocity','width','x','y']
    elif "maintain_velocity" in f:
        data.columns = ["Vid", "Object", "C1", "C2"]
        cluster_var = ['C1', 'C2']
        save_name = "_maintain_velocity"
        data_group_name = "_withVelocity.csv"
        data_cols = ['cog_x','cog_y','frame','height','obj','velocity','width','x','y']
    else:
        continue
    # Cluster
    X = np.asarray(data[cluster_var]).reshape((len(data[cluster_var[0]]), len(cluster_var)))
    clusters = hcluster.fclusterdata(X, 1.5, criterion="distance")
    # Save Cluster Plot and Cluster Results
    data_results = data
    data_results['Cluster'] = clusters 
    if len(cluster_var) == 1:
        plt.scatter(range(0, len(data['C'])) ,data['C'].values, c=clusters)
    else:
        plt.scatter(data['C1'].values, data['C2'].values, c=clusters)
    plt.savefig("cluster_plots/" + vid_id + save_name + ".png")
    # Find Smallest Cluster
    data_group = data[['Object', 'Cluster']].groupby(['Cluster']).agg(['count'])
    data_group['Cluster'] = data_group.index
    smallest_cluster = data_group.sort_values([('Object', 'count')])['Cluster'].iloc[0]
    # Get Object IDs
    anomaly_objs = data['Object'][data['Cluster'] == smallest_cluster]
    # Get X, Y, Height, Length, and Frame IDs
    anomaly_obj_data = pd.read_csv("data/" + str(vid_id) + data_group_name, header=None)
    anomaly_obj_data.columns = data_cols
    anomaly_obj_data = anomaly_obj_data[['obj', 'x', 'y', 'height', 'width', 'frame']]
    anomaly_obj_data = anomaly_obj_data[anomaly_obj_data['obj'].isin(anomaly_objs)]
    # Save File with Object ID, X, Y, Height, Length, Frame IDs
    anomaly_obj_data.to_csv("anomalies/" + str(vid_id) + save_name + ".csv")
    # Draws Bounding Boxes
    frames = anomaly_obj_data['frame'].unique()
    for i in frames: 
        image_name = str(vid_id) + "/" + str(i) + '.png'
	print image_name
        img = cv2.imread(image_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        anomaly_cars_in_frame = anomaly_obj_data[anomaly_obj_data['frame'] == int(i)]
        for index, row in anomaly_cars_in_frame.iterrows():
            x1 = row['x']
            y1 = row['y']
            h = row['height']
            w = row['width']
            cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (0, 0, 255), 3)
        labeled_image_name = "anomaly_pics/" + vid_id + "_" + str(i) + save_name + ".png"
        cv2.imwrite(labeled_image_name, img)
    



# In[ ]:



