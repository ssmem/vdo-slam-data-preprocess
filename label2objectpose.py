import os
import sys
import math
import numpy as np


def str2num(objstr):
    if objstr=="Car":
        return 3
    elif objstr=="Cyclist":
        return 2
    elif objstr=="Pedestrian":
        return 1
    elif objstr=="Tram":
        return 7
    elif objstr=="Person":
        return 1
    elif objstr=="Truck":
        return 8
    elif objstr=="Misc":
        return 4
    elif objstr=="Van":
        return 6



with open("object_pose.txt", 'w') as wf:
    with open("0020.txt", 'r') as f:
        for line in f.readlines():
            #line = line.strip('\n')
            ds = line.split()
            if ds[2]!="DontCare":
                wf.write(ds[0]+" "+str2num(ds[2])+" "+ds[6]+" "+ds[7]+" "+ds[8]+" "+ds[9]+" "+ds[13]+" "+ds[14]+" "+ds[15]+" "+ds[16]+"\n")




