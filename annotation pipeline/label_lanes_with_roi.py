import cv2
import numpy as np
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import sys
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

video_name = sys.argv[1]
lane_points = sys.argv[2]

sections = []
with open(lane_points) as f:
    for line in f:
        contour_points = []
        vals = line.split(") (")
        if vals[0] == "stop":
            break
        for i in vals:
            newstr = i.replace("(", "")
            newstr = newstr.replace(")", "")
            newstr = newstr.replace("\n", "")
            nums = newstr.split(", ")
            points = (float(nums[0]), float(nums[1]))
            contour_points.append(points)
        sections.append(contour_points)

#polygon = Polygon(sections[0])
#print polygon.contains(Point(1000, 990))

def checkLane(contour, point1, point2, prev_result):
    polygon = Polygon(contour)
    if polygon.contains(point1) and polygon.contains(point2):
        return 0
    else:
        return prev_result


# read in parsed video csv label cars in their lanes


# In[187]:

csv_file = "parsed_csv/" + video_name + "_byobject.csv"
cars_df = pd.read_csv(csv_file, index_col=0)

lane = []
new_df = None
cars_df['lane'] = 1
obj_temp_df = cars_df
for section in sections:
#    obj_temp_df['mid_points'] = obj_temp_df.apply((lambda x: (x['cog_x'], x['cog_y'])), axis=1)
#    obj_temp_df['points'] = obj_temp_df.apply((lambda x: (x['x'], x['y'])), axis=1)
    lane_results = obj_temp_df.apply(lambda x: checkLane(section, Point(x['cog_x'], x['cog_y']), Point(x['x'], x['y']), x['lane']), axis=1)
    obj_temp_df['lane'] = lane_results

new_df = obj_temp_df

# In[ ]:

new_df.to_csv("parsed_csv/" + video_name + "_withLanes.csv")
