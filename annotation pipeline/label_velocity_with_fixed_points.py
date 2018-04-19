
# coding: utf-8

# In[20]:

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
import math
import numpy as np
import cv2 
import sys

video_name = sys.argv[1]
video_folder = sys.argv[2]
video_points = sys.argv[3]

def getTransformPoints(angle, p, h=False):
    if not h:
        angle = angle - np.pi / 2.0
    t_x = np.cos(angle) * float(p[0]) + np.sin(angle) * float(p[1])
    t_y = -1 * np.sin(angle) * float(p[0]) + np.cos(angle) * float(p[1])
    if not h:
        return t_y
    return t_x


# In[ ]:

parsed_obj_csv = video_name + "_byobject"


# In[3]:

df = pd.read_csv("parsed_csv/" + parsed_obj_csv + ".csv", index_col = 0)

# estimate "velocity"
# for every object, calc dist(frame1(cog_x, cog_y), frame2(cog_x, cog_y))
# divide distance by "length" of car (avg of width or height) and multiple by avg car length
# ^ above is "distance" divide this by (# frames / fps)


# In[13]:

objs = df['obj'].unique()
frames = df['frame'].unique()
frames.sort()


# In[14]:

sections = []
with open(video_points) as f:
    for line in f:
        vals = line.split()
        if vals[0] == "Marks":
            continue
        if vals[0] == "stop" or vals[0] == "Lanes":
            break
        sections.append(vals)

new_df = None
for o in objs:
    obj_pos = df[df['obj'] == o]
    if len(obj_pos) < 10:
        continue
    obj_df = None
    for section in sections:
        if len(obj_pos) <= 0:
            break
        obj_temp_df = obj_pos
        if len(section) == 4:
            obj_temp_df['y_tf'] = obj_temp_df['cog_y'] - (float(section[3]) * obj_temp_df['cog_x'])
            obj_temp_df = obj_temp_df[(obj_temp_df['y_tf'] >= float(section[0])) & (obj_temp_df['y_tf'] <= float(section[1]))]
            del obj_temp_df['y_tf']
        elif len(section) == 5:
            if section[4] == "vertical":
                horizontal = False
            else:
                horizontal = True
            obj_temp_df['points'] = obj_temp_df.apply((lambda x: (x['cog_x'], x['cog_y'])), axis=1)
            obj_temp_df['tf'] = obj_temp_df['points'].apply(lambda x: getTransformPoints(float(section[3]), x, h=horizontal))
            obj_pos = obj_temp_df[(obj_temp_df['tf'] < float(section[0])) | (obj_temp_df['tf'] > float(section[1]))]
            obj_temp_df = obj_temp_df[(obj_temp_df['tf'] >= float(section[0])) & (obj_temp_df['tf'] <= float(section[1]))]
            obj_temp_df = obj_temp_df.drop(columns=['tf', 'points'])
            obj_pos = obj_pos.drop(columns=['tf', 'points'])
#            del obj_temp_df['tf']
#            del obj_temp_df['points']
        else:
            obj_temp_df = obj_pos[(obj_pos['cog_y'] >= float(section[0])) & (obj_pos['cog_y'] <= float(section[1]))]
        if len(obj_temp_df) > 0:
            v = float(section[2]) / (len(obj_temp_df) / 30.0)
        else:
            obj_temp_df = obj_pos
            v = -1  # cannot calc velocity for these objects in this section
        obj_temp_df['velocity'] = v
        if obj_df is None:
            obj_df = obj_temp_df
        else:
            obj_df = pd.concat([obj_df, obj_temp_df])

    obj_pos = obj_df
    nonneg = obj_pos['velocity'][obj_pos['velocity'] != -1]
    if len(nonneg) > 0:
        avg = nonneg.mean()
    else:
        avg = -1
    obj_pos['velocity'][obj_pos['velocity'] == -1] = avg
    if new_df is None:
        new_df = obj_pos
    else:
        new_df = pd.concat([new_df, obj_pos])


final_df = new_df
if 'tf' in final_df.columns:
    final_df = final_df.drop(columns=['tf'])
if 'points' in final_df.columns:
    final_df = final_df.drop(columns=['points'])
final_df = final_df[(final_df['velocity'] < 60) & (final_df['velocity'] >= 0)]


# In[152]:

final_df.to_csv("parsed_csv/" + video_name + "_withVelocity.csv")


# In[22]:

# formatting into df by frame
df = final_df
frames = df['frame'].unique()


# In[23]:

top = []
bottom = []
width = []
height = []
obj_ids = []
velocity = []
car_density = []
frame_number = []

for i in frames:
    temp = df[df['frame'] == i]
    top_string = ""
    bottom_string = ""
    width_string = ""
    height_string = ""
    obj_id_string = ""
    velocity_string = ""
    
    cars = 0
    for index, row in temp.iterrows():
        if top_string == "":
            obj_id_string = obj_id_string + str(row['obj'])
            top_string = top_string + str(row['x'])
            bottom_string = bottom_string + str(row['y'])
            width_string = width_string + str(row['width'])
            height_string = height_string + str(row['height'])
            velocity_string = velocity_string + str(row['velocity'])
        else:
            obj_id_string = obj_id_string + "," + str(row['obj'])
            top_string = top_string + "," + str(row['x'])
            bottom_string = bottom_string + "," + str(row['y'])
            width_string = width_string + "," + str(row['width'])
            height_string = height_string + "," + str(row['height'])
            velocity_string = velocity_string + "," + str(row['velocity'])
        cars = cars + 1

    top.append(top_string)
    bottom.append(bottom_string)
    width.append(width_string)
    height.append(height_string)
    obj_ids.append(obj_id_string)
    car_density.append(cars)
    frame_number.append(i)
    velocity.append(velocity_string)


# In[49]:

names = ["x", "y", "height", "width", "obj_ids", "num_cars", "frame", 'velocity']
output_df = pd.DataFrame.from_items(zip(names, [top, bottom, height, width, obj_ids, car_density, frame_number, velocity]))



# In[43]:

output_df.to_csv("parsed_csv/" + video_name + ".csv")


# In[52]:

# Add Velocity to Picture
#def closest_node(node, nodes):
#    nodes = np.asarray(nodes)
#    dist_2 = np.sum((nodes - node)**2, axis=1)
#    if min(dist_2) > 10000:
#        return None
#    return np.argmin(dist_2)
#
#
## In[72]:
#
#objects = {}
#object_points = []
#obj_id = 0
#for index, row in output_df.iterrows():
#    #print row['frame']
#
#    # path to image
#    frame_label=str(row['frame'])
#    image_name = video_folder + "/" + frame_label + '.png'
#    #print image_name
#    img=cv2.imread(image_name)
#    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#    num_cars = row['num_cars']
#    if num_cars <= 0:
#        continue
#
#    split_x = [float(a) for a in row['x'].split(",")]
#    split_y = [float(a) for a in row['y'].split(",")]
#    split_width = [float(a) for a in row['width'].split(",")]
#    split_height = [float(a) for a in row['height'].split(",")]
#    split_velocity = [float(a) for a in row['velocity'].split(",")]
#    frame = int(row['frame'])
#    num_cars = row['num_cars']
#
#    objs_to_check = object_points
#
#    for i in range(0, num_cars):
#        x1 = float(split_x[i])
#        y1 = float(split_y[i])
#
#        w = float(split_width[i])
#        h = float(split_height[i])
#
#        x2 = x1 + w
#        y2 = y1 + h
#
#        x_mid = (x1 + x2) / 2.0
#        y_mid = (y1 + y2) / 2.0
#
#        v = float(split_velocity[i])
#
#        if index == 0:
#            object_dict = {}
#            object_dict[frame] = {'x': x1,
#                                  'y': y1,
#                                  'width': w,
#                                  'height': h,
#                                  'cog_x': x_mid,
#                                  'cog_y': y_mid,
#                                  'velocity': v
#                                 }
#            objects[obj_id] = object_dict
#            object_points.append((x1, y1))
#
#            cv2.rectangle(img, (int(x1),int(y1)), (int(x1+w),int(y1+h)), (0, 0, 255), 3)
#            cv2.putText(img, str(round(v, 2)), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 5)
#            cv2.putText(img, str(round(v, 2)), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_PLAIN, 2, (25, 255, 255), 3)
#
#            obj_id += 1
#        else:
#            # check if object is the same as a labeled object
#            closest_object = closest_node((x1, y1), objs_to_check)
#            if closest_object is None:
#                object_dict = {}
#                object_dict[frame] = {'x': x1,
#                                      'y': y1,
#                                      'width': w,
#                                      'height': h,
#                                      'cog_x': x_mid,
#                                      'cog_y': y_mid,
#                                      'velocity': v
#                                 }
#                objects[obj_id] = object_dict
#                object_points.append((x1, y1))
#
#                cv2.rectangle(img, (int(x1),int(y1)), (int(x1+w),int(y1+h)), (0, 0, 255), 3)
#                cv2.putText(img, str(round(v, 2)), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 5)
#                cv2.putText(img, str(round(v, 2)), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_PLAIN, 2, (25, 255, 255), 3)
#
#                obj_id += 1
#            else:
#                if frame in objects[closest_object]:
#                    continue
#                objects[closest_object][frame] = {'x': x1,
#                                                  'y': y1,
#                                                  'width': w,
#                                                  'height': h,
#                                                  'cog_x': x_mid,
#                                                  'cog_y': y_mid,
#                                                  'velocity': v
#                                                 }
#                object_points[closest_object] = (x1, y1)
#                # keep this object from being checked again
#                # objs_to_check[closest_object] = (-999999, -999999)
#
#
#                cv2.rectangle(img, (int(x1),int(y1)), (int(x1+w),int(y1+h)), (0, 0, 255), 3)
#                cv2.putText(img, str(round(v, 2)), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 5)
#                cv2.putText(img, str(round(v, 2)), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_PLAIN, 2, (25, 255, 255), 3)
#
#    labeled_image_name = 'labeled_images/Velocity/' + video_name + "/" + frame_label + '.png'
#print labeled_image_name
#    cv2.imwrite(labeled_image_name,img)




