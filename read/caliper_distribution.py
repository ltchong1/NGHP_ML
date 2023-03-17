# -*- coding: utf-8 -*-
"""
@author: leeby
"""

import numpy
import matplotlib

numpy.random.seed(20221201)
matplotlib.rcParams['figure.dpi']=600

#well specific parameters
well=5
# upper & lower TVD cutoffs
cutoff=[[2943,3353],
        [2773,3148],
        [1998.8,2172],
        [2270,2501],
        [2398,9999],
        [1924.5,2069.5]]

#original NGHP-02-08-A_LWD_RM_0.5ft_ MD_LAS_2190m-2570m.las
converted=[]
file=open('NGHP2_'+str(well)+'_field_print.txt','r')
lines=file.readlines()
file.close()
for x in range(1,len(lines)):
    splitted=lines[x].split()
    floated=[float(y) for y in splitted]
    converted.append(floated)
FP=numpy.array(converted)

#original NGHP-02-08-A_8.5in_SonicScope_RKB.las
converted=[]
file=open('NGHP2_'+str(well)+'_sonicscope.txt','r')
lines=file.readlines()
file.close()
for x in range(1,len(lines)):
    splitted=lines[x].split()
    floated=[float(y) for y in splitted]
    converted.append(floated)
SS=numpy.array(converted)

#original 20210902-Sgh_morph_ass.xlsx
converted=[]
file=open('NGHP2_'+str(well)+'_morph.txt','r')
lines=file.readlines()
file.close()
for x in range(1,len(lines)):
    splitted=lines[x].split()
    converted.append(splitted)
morph=numpy.array(converted)

#TVD GR RES_BD RHOB DCAV VP Sgh morph
matched=numpy.concatenate((FP[:,[2,5,26,31,35]],numpy.zeros((len(FP),3))),axis=1)

#match sonicsope depths to field print depths
for x in range(len(FP[:,2])):
    diff=[abs(y-FP[x,2]) for y in SS[:,0]]
    if min(diff)<=.1524:
        matched[x,5]=SS[diff.index(min(diff)),-3]
    else:
        matched[x,5]=-9999
        print('sonicscope depth deviated',min(diff),'@',FP[x,2])

#add morph to matched array
floated=[float(x) for x in morph[:,0]]
for x in range(len(FP[:,2])):
    diff=[abs(y-FP[x,2]) for y in floated]
    if min(diff)<=.1524:
        matched[x,6]=float(morph[diff.index(min(diff)),1])
        if morph[diff.index(min(diff)),2].lower()=='f':
            matched[x,7]=1
        elif morph[diff.index(min(diff)),2].lower()=='p':
            matched[x,7]=2
    else:
        print('skipping : morph depth deviated',min(diff),'@',FP[x,2])

#filter out non-data & data outside cutoff
filtered=[]
for x in range(len(matched)):
    if (-999.25 not in matched[x,:]) and (-9999 not in matched[x,:]) and matched[x,0]>cutoff[well-5][0] and matched[x,0]<cutoff[well-5][1]:
        filtered.append(matched[x,:])
filtered=numpy.array(filtered)

matplotlib.pyplot.hist(filtered[:,4],50)
matplotlib.pyplot.title('site '+str(well))
matplotlib.pyplot.xlabel('DCAV')
matplotlib.pyplot.ylabel('count')
print('total data points =',len(filtered))
print('95th percentile =',numpy.percentile(filtered[:,4],95))
