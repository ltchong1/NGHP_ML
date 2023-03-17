# -*- coding: utf-8 -*-
"""
@author: leeby
"""

import os
import numpy
import sklearn.model_selection
from tensorflow import keras
import keras_tuner

directory='C:/Users/leeby/Documents/work/NETL/hydrate/python/NGHP/saturation_tuning'
os.chdir(directory)
log_dir='/logs/KT_log_1/'

#settings
NGHP_wells=[5,6,7,8,9,10]
split=.2
act_fun='relu'
max_try=10
bat_size=100
eps=500

#initialize
depth=[]
density=[]
porosity=[]
GR=[]
Rt=[]
Vp=[]
Vs=[]
Sgh=[]
morph=[]

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
        GR[-1].append(floated[1])
        Rt[-1].append(floated[2])
        density[-1].append(floated[3])
        porosity[-1].append(floated[4])
        Vp[-1].append(floated[5])
        Sgh[-1].append(floated[6])
        morph[-1].append(floated[7])

GR_1D=[x for y in GR for x in y]
Rt_1D=[x for y in Rt for x in y]
density_1D=[x for y in density for x in y]
porosity_1D=[x for y in porosity for x in y]
Vp_1D=[x for y in Vp for x in y]
Sgh_1D=[x for y in Sgh for x in y]
morph_1D=[x for y in morph for x in y]

#well log combination
X_1D=numpy.transpose([GR_1D,Rt_1D,density_1D,porosity_1D,Vp_1D,morph_1D])
X=[]
for x in range(len(depth)):
    X.append([])
    for y in range(len(depth[x])):
        X[-1].append([GR[x][y],Rt[x][y],density[x][y],porosity[x][y],Vp[x][y]])

#shuffle & split
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(X_1D,numpy.array(Sgh_1D),test_size=split)
test_morphs=x_test[:,-1]
x_train=numpy.delete(x_train,-1,axis=1)
x_test=numpy.delete(x_test,-1,axis=1)

#scale
transformer=sklearn.preprocessing.MinMaxScaler().fit(x_train)
x_train_norm=transformer.transform(x_train)
x_test_norm=transformer.transform(x_test)
X_norm=[]
for y in range(len(X)):
    X_norm.append(transformer.transform(X[y]))

def build_model(HP):
    keras.backend.clear_session()
    
    #hyperparameter ranges
    HP_units1=HP.Int('units1',min_value=50,max_value=175,step=25)
    HP_drop1=HP.Float('drop1',min_value=.1,max_value=.2,step=.1)
    HP_units2=HP.Int('units2',min_value=50,max_value=175,step=25)
    HP_drop2=HP.Float('drop2',min_value=.1,max_value=.2,step=.1)
    HP_learn=HP.Choice('learn',[.0001,.0005,.001])
    
    #construct neural network
    input1=keras.Input(shape=x_train_norm[0].shape,name='well_log_data')
    NN=keras.layers.Dense(units=HP_units1,activation=act_fun)(input1)
    NN=keras.layers.Dropout(HP_drop1)(NN)
    NN=keras.layers.Dense(units=HP_units2,activation=act_fun)(NN)
    NN=keras.layers.Dropout(HP_drop2)(NN)
    output1=keras.layers.Dense(units=1,activation='linear',name='gas_hydrate_saturation')(NN)
    NN=keras.Model(inputs=input1,outputs=output1)
    
    #neural network options & compilation
    NN.compile(optimizer=keras.optimizers.Adam(learning_rate=HP_learn),loss=keras.losses.MeanSquaredError(),metrics=[keras.metrics.MeanAbsoluteError()])
    return NN

#tuning
tuner=keras_tuner.Hyperband(build_model,objective='val_loss',max_epochs=eps,overwrite=True,directory='C:/Users/leeby/Documents/work/NETL/hydrate/python/temp9b')
tuner.search(x_train_norm,y_train,batch_size=bat_size,epochs=eps,validation_data=(x_test_norm,y_test),callbacks=[keras.callbacks.TensorBoard('C:/Users/leeby/Documents/work/NETL/hydrate/python/temp9b')],verbose=0)
tuner.results_summary()
