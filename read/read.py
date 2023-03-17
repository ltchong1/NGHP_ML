# -*- coding: utf-8 -*-
"""
@author: ctlek_000
"""

import os
import numpy
from loop import LoOP

directory='C:/Users/leeby/Documents/work/NETL/hydrate/python/NGHP/read'
os.chdir(directory)

#well specific parameters
well='10'
mat_den=2.7 #2019 collett et al table 3 matrix densities by well : 5 = 2.69 , 6 = 2.69 , 7 = 2.71 , 8 = 2.67 , 9 = 2.7 , 10 = 2.7
cutoff=[1924.5,2069.5] # upper & lower TVD cutoffs

#original NGHP-02-08-A_LWD_RM_0.5ft_ MD_LAS_2190m-2570m.las
converted=[]
file=open('NGHP2_'+well+'_field_print.txt','r')
lines=file.readlines()
file.close()
for x in range(1,len(lines)):
    splitted=lines[x].split()
    floated=[float(y) for y in splitted]
    converted.append(floated)
FP=numpy.array(converted)

#original NGHP-02-08-A_8.5in_SonicScope_RKB.las
converted=[]
file=open('NGHP2_'+well+'_sonicscope.txt','r')
lines=file.readlines()
file.close()
for x in range(1,len(lines)):
    splitted=lines[x].split()
    floated=[float(y) for y in splitted]
    converted.append(floated)
SS=numpy.array(converted)

#original 20210902-Sgh_morph_ass.xlsx
converted=[]
file=open('NGHP2_'+well+'_morph.txt','r')
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
numpy.savetxt('matched.txt',matched,fmt='%.9f',delimiter='\t')

#counting non-data
bad=numpy.zeros(len(matched[0,:]))
bad[0]=numpy.count_nonzero(matched[:,0]==-999.25)
bad[1]=numpy.count_nonzero(matched[:,1]==-999.25)
bad[2]=numpy.count_nonzero(matched[:,2]==-999.25)
bad[3]=numpy.count_nonzero(matched[:,3]==-999.25)
bad[4]=numpy.count_nonzero(matched[:,4]==-999.25)
bad[5]=numpy.count_nonzero(matched[:,5]==-9999)

#filter out non-data & data deeper than cutoff
filtered=[]
for x in range(len(matched)):
    if (-999.25 not in matched[x,:]) and (-9999 not in matched[x,:]) and matched[x,0]>cutoff[0] and matched[x,0]<cutoff[1]:
        filtered.append(matched[x,:])
filtered=numpy.array(filtered)
numpy.savetxt('filtered.txt',filtered,fmt='%.9f',delimiter='\t')

#caliper outlier removal
upper=numpy.percentile(filtered[:,4],95)
no_cal_out=[]
cal_out=[]
for x in range(len(filtered)):
    if filtered[x,4]<upper:
        no_cal_out.append(filtered[x])
    else:
        cal_out.append(filtered[x])
no_cal_out=numpy.array(no_cal_out)
cal_out=numpy.array(cal_out)
numpy.savetxt('cal_out.txt',cal_out,fmt='%.9f',delimiter='\t')

#replace caliper column with density porosity calculation
for x in range(len(no_cal_out)):
    no_cal_out[x,4]=(mat_den-no_cal_out[x,3])/(mat_den-1.05)
numpy.savetxt('no_cal_out.txt',no_cal_out,fmt='%.9f',delimiter='\t')
#TVD GR RES_BD RHOB den_por VP Sgh morph

#GLOSS outlier removal not including depth , Sgh , & morphs
del_feat=numpy.delete(no_cal_out,[0,6,7],axis=1)
soup=LoOP(del_feat)
soup_init=soup.local_outlier_probabilities(verbose=False,feature_end=-1,L=2,k=20)
soup_window_scores=[]
for window in range(len(del_feat[0])-1):
    soup_scores=soup.local_outlier_search(feature_start=window,feature_end=window+2,L=2,k=20)
    soup_window_scores.append(soup_scores)
soup_window_scores.append(soup_init)
soup_window_scores=numpy.array(soup_window_scores)
soup_scores=soup_window_scores.max(axis=0)
outliers=list(numpy.where(soup_scores>.99)[0])
no_out=numpy.delete(no_cal_out,outliers,0)
numpy.savetxt('data'+well+'.txt',no_out,fmt='%.9f',delimiter='\t')
numpy.savetxt('soup.txt',soup_window_scores,fmt='%.9f',delimiter='\t')
numpy.savetxt('out.txt',outliers,fmt='%.0f',delimiter='\t')

#histogram
Sgh=[[],[],[]]
porosity=[[],[],[]]
for x in range(len(no_out)):
    Sgh[int(no_out[x,7])].append(no_out[x,6])
    porosity[int(no_out[x,7])].append(no_out[x,4])
Sgh_hist=[[],[],[]]
porosity_hist=[[],[],[]]
bin_range=numpy.arange(0,1.01,.01)
for x in range(3):
    Sgh_hist[x],bin_edges=numpy.histogram(Sgh[x],bin_range)
    porosity_hist[x],bin_edges=numpy.histogram(porosity[x],bin_range)
numpy.savetxt('Sgh_hist.txt',numpy.transpose(Sgh_hist),fmt='%.0f',delimiter='\t')
numpy.savetxt('porosity_hist.txt',numpy.transpose(porosity_hist),fmt='%.0f',delimiter='\t')
