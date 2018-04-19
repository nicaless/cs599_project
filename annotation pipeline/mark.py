import cv2
import sys
import numpy as np

filename = sys.argv[1]
savefile = sys.argv[2]

points=[]
mark_points = []
lane_points = []
lane_vals = []
cropping=False
init_point = []
lengths = []
horizontal=False
lane_num = 0


def click(event,x,y,flags,param):
    global points,cropping,init_point
    if event==cv2.EVENT_LBUTTONDOWN:
        init_point=[(x,y)]
        cropping=True
        #print(init_point)

    elif event==cv2.EVENT_LBUTTONUP:
        init_point.append((x,y))
        cropping=False
        cv2.line(img,init_point[0],init_point[1],(255,0,0),2)
        cv2.imshow("img",img)
        #print(init_point)
        cv2.line(img,init_point[0],(1920,init_point[0][1]),(0,255,0),2)
        cv2.line(img,init_point[1],(1920,init_point[1][1]),(0,255,0),2)
        points.append(init_point)
        init_point = []


img=cv2.imread(filename)
clone=img.copy()
cv2.namedWindow("img")
cv2.setMouseCallback("img",click)



while 1:
    cv2.imshow("img",img)
    key=cv2.waitKey(1) & 0xFF
    
    if key==ord("t"):
        print "setting to transform to horizontal"
        horizontal=True

    if key==ord("r"):
        print "reset"
        img=clone.copy()
        points=[]
        lengths=[]

    if key==ord("u"):
        print "undo"
        img=clone.copy()
        points=points[:-1]

    if key==ord("f"):
        print "reset lane marking"
        img=clone.copy()
        lane_vals = []
        lane_num = 0
        points=[]
    if key==ord("c"):
        print "clear markings but keep previous points"
        img=clone.copy()
    
    if key==ord("w"):
        print "marking white lane"
        lengths.append(3.048)

    if key==ord("s"):
        print "marking spaces"
        lengths.append(9.144)
    if key==ord("h"):
        print "marking half space"
        lengths.append(4.572)

    if key==ord("l"):
        print "marking lanes"
        mark_points = points
        points = []
        img=clone.copy()

    if key==ord("o"):
        print "marking turn out lane"
        lane_vals.append(-1)
    if key==ord("i"):
        print "marking good lane"
        lane_vals.append(lane_num)
        lane_num = lane_num + 1
    if key==ord("k"):
        print "keep previous lane val"
        lane_vals.append(lane_vals[len(lane_vals)-1])

    if key==ord("q"):
        lane_points = points
        break

print mark_points
print lane_points


def calcSlopeAndIntercept(p1, p2):
    m = (float(p2[1]) - float(p1[1])) / (float(p2[0]) - float(p2[1]))
    intercept = p2[1] - m * p2[0]
    return m, intercept

def getAngle(p1, p2):
    angles = np.arctan2([p2[1] - p1[1]], [p2[0] - p1[0]])
    return angles

def getTransformPoints(angle, p, h=False):
    if not h:
        angle = angle - np.pi / 2.0
    t_x = np.cos(angle) * float(p[0]) + np.sin(angle) * float(p[1])
    t_y = -1 * np.sin(angle) * float(p[0]) + np.cos(angle) * float(p[1])
    if not h:
        return t_y
    return t_x


# format lane markings
print("format lane markings")
out = []
for i, p in enumerate(mark_points):
    angle = getAngle(p[0], p[1])
    p0_t = getTransformPoints(angle, p[0], h=horizontal)
    p1_t = getTransformPoints(angle, p[1], h=horizontal)
    if horizontal:
        tilt = "horizontal"
    else:
        tilt = "vertical"
    out_line = [p0_t[0], p1_t[0], lengths[i], angle[0], tilt]
    out.append(out_line)

# format lane lines
print("format lane lines")
horizontal = not horizontal
lanes = []
for i, p in enumerate(lane_points):
    angle = getAngle(p[0], p[1])
    p0_t = getTransformPoints(angle, p[0], h=horizontal)
    p1_t = getTransformPoints(angle, p[1], h=horizontal)
    if horizontal:
        tilt = "horizontal"
    else:
        tilt = "vertical"
    out_line = [p0_t[0], p1_t[0], lane_vals[i], angle[0], tilt]
    lanes.append(out_line)


# write outfile

print("write outfile")
with open(savefile, "w") as f:
    f.write("Marks")
    f.write("\n")
    for o in out:
        for i in range(len(o)):
            f.write(str(o[i]))
            if i != (len(o) - 1):
                f.write(" ")
        f.write("\n")
    f.write("Lanes")
    f.write("\n")
    for l in lanes:
        for i in range(len(l)):
            f.write(str(l[i]))
            if i != (len(l) - 1):
                f.write(" ")
        f.write("\n")
    f.write("stop")



