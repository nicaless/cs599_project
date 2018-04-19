import cv2
import sys
import numpy as np
from roipoly import roipoly
import pylab as pl
import copy

filename = sys.argv[1]
savefile = sys.argv[2]
num_rois = int(sys.argv[3])
rois = []
colors = ['r', 'b', 'g']


img = pl.imread(filename)
clone=img.copy()
pl.imshow(img)
pl.colorbar()
ROI1 = roipoly(roicolor='r') # draw new ROI in red color
rois.append(ROI1)
print ROI1.allxpoints
print ROI1.allypoints
pl.imshow(img, interpolation='nearest', cmap="Greys")
ROI1.displayROI()
#ROI2 = roipoly(roicolor='b') # draw new ROI in red color
#rois.append(ROI2)
#print ROI2.allxpoints
#print ROI2.allypoints

print rois

with open(savefile, "w") as f:
    for roi in rois:
        for i in range(0, len(roi.allxpoints)):
            p = str((roi.allxpoints[i], roi.allypoints[i]))
            f.write(p)
            if i < len(roi.allxpoints) -1:
                f.write(" ")
        f.write("\n")
    f.write("stop")




