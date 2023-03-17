# -*- coding: utf-8 -*-
"""
@author: leeby
"""

import os
import numpy
import sklearn.model_selection
from tensorflow import keras

directory='C:/Users/leeby/Documents/work/NETL/hydrate/python/NGHP/saturation'
os.chdir(directory)

#hyperparameters
runs=100
split=.2
act_fun='relu'
net_top=[425,350]
drop_rate=[.1,.1]
learn_rate=.0015
bat_size=100
eps=500
ANSMallik_wells=['MTE_no_out.txt','IGS_no_out.txt','BP_no_out.txt','5L38_no_out.txt','2L38_no_out.txt']
NGHP_wells=[5,6,7,8,9,10]

depth=[]
density=[]
porosity=[]
GR=[]
Rt=[]
Vp=[]
Vs=[]
Sgh=[]
morph=[]

#open & read well log data for ANS & mallik
for x in ANSMallik_wells:
    file=open(x,'r')
    lines=file.readlines()
    file.close()
    depth.append([])
    density.append([])
    porosity.append([])
    GR.append([])
    Rt.append([])
    Vp.append([])
    Vs.append([])
    Sgh.append([])
    morph.append([])
    for y in range(len(lines)):
        splitted=lines[y].split()
        floated=[float(z) for z in splitted]
        depth[-1].append(floated[0])
        density[-1].append(floated[1])
        porosity[-1].append(floated[2])
        GR[-1].append(floated[3])
        Rt[-1].append(floated[4])
        Vp[-1].append(floated[5])
        Vs[-1].append(floated[6])
        Sgh[-1].append(floated[7])

#open & read well log data for NGHP
for x in NGHP_wells:
    file=open('data'+str(x)+'.txt','r')
    lines=file.readlines()
    file.close()
    depth.append([])
    density.append([])
    porosity.append([])
    GR.append([])
    Rt.append([])
    Vp.append([])
    Vs.append([])
    Sgh.append([])
    morph.append([])
    for y in range(len(lines)):
        splitted=lines[y].split()
        floated=[float(z) for z in splitted]
        depth[-1].append(floated[0])
        density[-1].append(floated[3])
        porosity[-1].append(floated[4])
        GR[-1].append(floated[1])
        Rt[-1].append(floated[2])
        Vp[-1].append(floated[5])
        Sgh[-1].append(floated[6])
        morph[-1].append(floated[7])

#flatten data
depth_flat=[x for y in depth for x in y]
density_flat=[x for y in density for x in y]
porosity_flat=[x for y in porosity for x in y]
GR_flat=[x for y in GR for x in y]
Rt_flat=[x for y in Rt for x in y]
Vp_flat=[x for y in Vp for x in y]
Sgh_flat=[x for y in Sgh for x in y]

#well log combination
X_3D=[]
for x in range(len(depth)):
    X_3D.append([])
    for y in range(len(depth[x])):
        X_3D[-1].append([density[x][y],porosity[x][y],GR[x][y],Rt[x][y],Vp[x][y]])
        #X_3D[-1].append([GR[x][y],Rt[x][y],density[x][y],porosity[x][y]])
X_1D=numpy.transpose([density_flat,porosity_flat,GR_flat,Rt_flat,Vp_flat])
#X_1D=numpy.transpose([GR_flat,Rt_flat,density_flat,porosity_flat])

#multiple runs with random splitting
scores=[]
well_scores=[]
MSE_MAE=[]
best=0
for x in range(runs):
    #shuffle & split
    x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(X_1D,Sgh_flat,test_size=split)
    
    #scale
    transformer=sklearn.preprocessing.MinMaxScaler().fit(x_train)
    x_train_norm=transformer.transform(x_train)
    x_test_norm=transformer.transform(x_test)
    X_3D_norm=[]
    for y in range(len(X_3D)):
        X_3D_norm.append(transformer.transform(X_3D[y]))

    #construct neural network
    keras.backend.clear_session()
    input1=keras.Input(shape=x_train_norm[0].shape,name='well_log_data')
    NN=keras.layers.Dense(units=net_top[0],activation=act_fun)(input1)
    NN=keras.layers.Dropout(drop_rate[0])(NN)
    for y in range(1,len(net_top)):
        NN=keras.layers.Dense(units=net_top[y],activation=act_fun)(NN)
        NN=keras.layers.Dropout(drop_rate[y])(NN)
    output1=keras.layers.Dense(units=1,activation='linear',name='gas_hydrate_saturation')(NN)
    NN=keras.Model(inputs=input1,outputs=output1)
    
    #neural network options & compilation
    NN.compile(optimizer=keras.optimizers.Adam(learning_rate=learn_rate),loss=keras.losses.MeanSquaredError(),metrics=[keras.metrics.MeanAbsoluteError()])
    
    #train & test
    fitted=NN.fit(x_train_norm,numpy.array(y_train),batch_size=bat_size,epochs=eps,verbose=0)
    evaluated=NN.evaluate(x_test_norm,numpy.array(y_test),verbose=0)
    loss=fitted.history['loss']
    
    #predict & calculate R2
    y_pred=NN.predict(x_test_norm,verbose=0).flatten()
    R2=sklearn.metrics.r2_score(y_test,y_pred)
    scores.append(R2)
    MSE_MAE.append(evaluated)
    print('run',x+1)
    print('loss =',evaluated[0])
    print('R2 =',R2)
    
    #predict & calculate R2 for each well
    well_preds=[]
    well_scores.append([])
    for y in range(len(X_3D_norm)):
        well_preds.append(NN.predict(X_3D_norm[y],verbose=0).flatten())
        well_scores[-1].append(sklearn.metrics.r2_score(Sgh[y],well_preds[y]))
    print('R2 (wells) =',well_scores[-1])
                
    #save if the best R2 so far
    if R2>scores[best]:
        NN.save(directory+'/saved')
        best=x
        best_transformer=transformer
        best_preds=well_preds
        numpy.savetxt('loss.txt',loss,fmt='%.9f',delimiter='\t')

#output scores for all runs & well predictions for the best run
numpy.savetxt('scores.txt',scores,fmt='%.9f',delimiter='\t')
numpy.savetxt('well_scores.txt',well_scores,fmt='%.9f',delimiter='\t')
numpy.savetxt('MSE_MAE.txt',MSE_MAE,fmt='%.9f',delimiter='\t')
max_len=max([len(x) for x in best_preds])
with open('predictions.txt','w') as file:
    for x in range(max_len):
        for y in range(len(best_preds)):
            if x>=len(best_preds[y]):
                file.writelines('\t\t\t')
            else:
                file.writelines('%s\t' % depth[y][x])
                file.writelines('%s\t' % Sgh[y][x])
                file.writelines('%s\t' % best_preds[y][x])
        file.writelines('\n')
print('best run',best+1)
print('best MSE =',MSE_MAE[best][0])
print('best R2 =',scores[best])
print('best R2 (wells) =',well_scores[best])
