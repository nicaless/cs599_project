# coding: utf-8

# In[1]:

import os
import cv2
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import argparse
import sys
import time

video_name = sys.argv[1]
video_folder = sys.argv[2]

max_distance = 75
frame_limit = 5
if video_name == "2":
    max_distance = 170
if video_name == "49":
    max_distance = 100
    frame_limit = 10
if video_name == "74":
    max_distance = 170
    frame_limit = 10


# In[2]:

#def closest_node(node, nodes):
#    nodes = np.asarray(nodes)
#    dist_2 = np.sum((nodes - node) ** 2, axis=1)
#    if min(dist_2) > 10000:
#        return None
#    return np.argmin(dist_2)

def closest_node(node, nodes):
    obj_ids = nodes.keys()
    nodes = np.asarray(nodes.values())
    dist_2 = map(lambda x: np.linalg.norm(x-node), nodes)
    if min(dist_2) > max_distance:
        return None
    return obj_ids[np.argmin(dist_2)]



# In[3]:

# dictionary where key is object id and value is a dictionary containing key_values for
# video, frame, x,y,width,height, cog_x, cog_y, (and velocity, eventually)
objects = {}
#object_points = []
object_points = {}
obj_id = 0

# In[4]:

# video_name = "Loc1_1"
csv_file = pd.read_csv("csv/" + video_name + ".csv", index_col=0)
csv_file.columns = ['x', 'y', 'width', 'height', 'num_cars', 'frame']

# In[5]:

csv_file = csv_file.sort_values(['frame'])

# In[6]:

for index, row in csv_file.iterrows():
    if int(row['frame']) % 500 == 0:
        print str(row['frame']) + str(time.time())
    # print row['frame']

    # path to image
    frame_label = str(row['frame'])
    image_name = video_folder + "/" + frame_label + '.png'
    # print image_name
    img = cv2.imread(image_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    num_cars = row['num_cars']
    if num_cars <= 0:
        continue

    split_x = [float(a) for a in row['x'].split(",")]
    split_y = [float(a) for a in row['y'].split(",")]
    split_width = [float(a) for a in row['width'].split(",")]
    split_height = [float(a) for a in row['height'].split(",")]
    frame = int(row['frame'])

    objs_to_check = object_points

    for i in range(0, num_cars):
        x1 = float(split_x[i])
        y1 = float(split_y[i])

        w = float(split_width[i])
        h = float(split_height[i])

        x2 = x1 + w
        y2 = y1 + h

        x_mid = (x1 + x2) / 2.0
        y_mid = (y1 + y2) / 2.0

        if index == 0 or len(objs_to_check) == 0:
            object_dict = {}
            object_dict[frame] = {'x': x1,
                                  'y': y1,
                                  'width': w,
                                  'height': h,
                                  'cog_x': x_mid,
                                  'cog_y': y_mid
                                  }
            objects[obj_id] = object_dict
            #object_points.append((x_mid, y_mid))
            #object_points.append((x_mid, y_mid, x1, y1))
            object_points[obj_id] = (x_mid, y_mid, x1, y1)

            cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (0, 0, 255), 3)
            cv2.putText(img, str(obj_id), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 5)
            cv2.putText(img, str(obj_id), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_PLAIN, 2, (25, 255, 255), 3)

            obj_id += 1
        else:
            # check if object is the same as a labeled object
            #closest_object = closest_node((x_mid, y_mid), objs_to_check)
            closest_object = closest_node((x_mid, y_mid, x1, y1), objs_to_check)
            # if cannot find a close previous labeled object, create new object
            if closest_object is None:
                object_dict = {}
                object_dict[frame] = {'x': x1,
                                      'y': y1,
                                      'width': w,
                                      'height': h,
                                      'cog_x': x_mid,
                                      'cog_y': y_mid
                                      }
                objects[obj_id] = object_dict
                #object_points.append((x_mid, y_mid))
                #object_points.append((x_mid, y_mid, x1, y1))
                object_points[obj_id] = (x_mid, y_mid, x1, y1)

                cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (0, 0, 255), 3)
                cv2.putText(img, str(obj_id), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 5)
                cv2.putText(img, str(obj_id), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_PLAIN, 2, (25, 255, 255), 3)

                obj_id += 1
            else:
                # do not label duplicate
                # create new object if last time the 'closest object' was seen is many frames ago or 'closest object' was already labeled:
                if frame in objects[closest_object] or frame - max(objects[closest_object]) > frame_limit:
                    if frame - max(objects[closest_object]) > frame_limit:
                        #objs_to_check.remove(closest_object)
                        del objs_to_check[closest_object]
                    object_dict = {}
                    object_dict[frame] = {'x': x1,
                                          'y': y1,
                                          'width': w,
                                          'height': h,
                                          'cog_x': x_mid,
                                          'cog_y': y_mid
                                          }
                    objects[obj_id] = object_dict
                    #object_points.append((x_mid, y_mid))
                    #object_points.append((x_mid, y_mid, x1, y1))
                    object_points[obj_id] = (x_mid, y_mid, x1, y1)


                    cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (0, 0, 255), 3)
                    cv2.putText(img, str(obj_id), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 5)
                    cv2.putText(img, str(obj_id), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_PLAIN, 2, (25, 255, 255), 3)

                    obj_id += 1
                    continue

                objects[closest_object][frame] = {'x': x1,
                                                  'y': y1,
                                                  'width': w,
                                                  'height': h,
                                                  'cog_x': x_mid,
                                                  'cog_y': y_mid
                                                  }
                #object_points[closest_object] = (x_mid, y_mid)
                object_points[closest_object] = (x_mid, y_mid, x1, y1)

                cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (0, 0, 255), 3)
                cv2.putText(img, str(closest_object), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 5)
                cv2.putText(img, str(closest_object), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_PLAIN, 2, (25, 255, 255), 3)

    labeled_image_name = 'labeled_images/Numbered/' + video_name + '/' + frame_label + '.png'
    # print labeled_image_name
    cv2.imwrite(labeled_image_name, img)

# In[109]:

# Combining into a df
object_dfs = []
for i in objects:
    temp_df = pd.DataFrame.from_dict(objects[i], orient="index")
    temp_df['obj'] = i
    temp_df['frame'] = temp_df.index
    object_dfs.append(temp_df)

df = pd.concat(object_dfs)
df.to_csv("parsed_csv/" + video_name + "_byobject.csv")

# In[13]:

# formatting into df by frame
frames = df['frame'].unique()

# In[31]:

top = []
bottom = []
width = []
height = []
obj_ids = []
car_density = []
frame_number = []
for i in frames:
    temp = df[df['frame'] == i]
    top_string = ""
    bottom_string = ""
    width_string = ""
    height_string = ""
    obj_id_string = ""
    cars = 0
    for index, row in temp.iterrows():
        if top_string == "":
            obj_id_string = obj_id_string + str(row['obj'])
            top_string = top_string + str(row['x'])
            bottom_string = bottom_string + str(row['y'])
            width_string = width_string + str(row['width'])
            height_string = height_string + str(row['height'])
        else:
            obj_id_string = obj_id_string + "," + str(row['obj'])
            top_string = top_string + "," + str(row['x'])
            bottom_string = bottom_string + "," + str(row['y'])
            width_string = width_string + "," + str(row['width'])
            height_string = height_string + "," + str(row['height'])
        cars = cars + 1

    top.append(top_string)
    bottom.append(bottom_string)
    width.append(width_string)
    height.append(height_string)
    obj_ids.append(obj_id_string)
    car_density.append(cars)
    frame_number.append(i)

# In[32]:

names = ["top", "bottom", "height", "width", "obj_ids", "car_density", "frame_number"]
output_df = pd.DataFrame.from_items(zip(names, [top, bottom, height, width, obj_ids, car_density, frame_number]))

# In[33]:

output_df.to_csv("parsed_csv/" + video_name + ".csv")




