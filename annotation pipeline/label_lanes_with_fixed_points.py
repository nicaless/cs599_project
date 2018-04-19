import cv2
import numpy as np
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import sys

video_name = sys.argv[1]
lane_points = sys.argv[2]

def getTransformPoints(angle, p, h=False):
    if not h:
        angle = angle - np.pi / 2.0
    t_x = np.cos(angle) * float(p[0]) + np.sin(angle) * float(p[1])
    t_y = -1 * np.sin(angle) * float(p[0]) + np.cos(angle) * float(p[1])
    if not h:
        return t_y
    return t_x


sections = []
with open(lane_points) as f:
    lanes = False
    for line in f:
        vals = line.split()
        if vals[0] == "Lanes":
            lanes = True
            continue
        if vals[0] == "stop":
            break
        if lanes:
            sections.append(vals)

# read in parsed video csv label cars in their lanes


# In[187]:

csv_file = "parsed_csv/" + video_name + "_byobject.csv"
cars_df = pd.read_csv(csv_file, index_col=0)

lane = []
new_df = None
for section in sections:
    # transpose all points
    if section[4] == "vertical":
        horizontal = False
    else:
        horizontal = True
    obj_temp_df = cars_df

    obj_temp_df['mid_points'] = obj_temp_df.apply((lambda x: (x['cog_x'], x['cog_y'])), axis=1)
    obj_temp_df['points'] = obj_temp_df.apply((lambda x: (x['x'], x['y'])), axis=1)
    obj_temp_df['tf_mid'] = obj_temp_df['mid_points'].apply(lambda x: getTransformPoints(float(section[3]), x, h=horizontal))
    obj_temp_df['tf'] = obj_temp_df['points'].apply(lambda x: getTransformPoints(float(section[3]), x, h=horizontal))

    # find all cars with points within the section
#obj_temp_df_subset = obj_temp_df[(obj_temp_df['tf'] >= float(section[0])) & (obj_temp_df['tf'] <= float(section[1]))]
    obj_temp_df_subset = obj_temp_df[(obj_temp_df['tf'] >= float(section[0])) & (obj_temp_df['tf'] <= float(section[1])) & (obj_temp_df['tf_mid'] >= float(section[0])) & (obj_temp_df['tf_mid'] <= float(section[1]))]
#cars_df = obj_temp_df[(obj_temp_df['tf'] < float(section[0])) | (obj_temp_df['tf'] > float(section[1]))]
    cars_df = obj_temp_df[(obj_temp_df['tf'] < float(section[0])) | (obj_temp_df['tf'] > float(section[1])) & (obj_temp_df['tf_mid'] < float(section[0])) | (obj_temp_df['tf_mid'] > float(section[1]))]
    del cars_df['tf']
    del cars_df['points']
    if len(obj_temp_df_subset) > 0:
        obj_temp_df_subset['lane'] = int(section[2])
        if new_df is None:
            new_df = obj_temp_df_subset
        else:
            new_df = pd.concat([new_df, obj_temp_df_subset])

if len(cars_df) > 0:
    cars_df['lane'] = -2
    new_df = pd.concat([new_df, cars_df])

# In[ ]:

new_df.to_csv("parsed_csv/" + video_name + "_withLanes.csv")
