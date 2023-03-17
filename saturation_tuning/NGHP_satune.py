# -*- coding: utf-8 -*-
"""
@author: leeby
"""

import os
import numpy
import sklearn.model_selection
from tensorflow import keras
from tensorflow import summary
from tensorboard.plugins.hparams import api

directory='C:/Users/leeby/Documents/work/NETL/hydrate/python/NGHP/saturation_tuning'
os.chdir(directory)
log_dir='logs/log_0/'

#hyperparameters
runs_per_param=10
split=.16667
act_fun='relu'
Nlayers=2
NGHP_wells=[5,6,7,8,9,10]

#hyperparameter tuning
HP_LEARN=api.HParam('learning rate',api.Discrete([.0001,.0003,.0005]))
HP_NODES=api.HParam('nodes per layer',api.Discrete([40,80,160]))
HP_DROPOUT=api.HParam('dropout rate',api.Discrete([.1,.3,.5]))
HP_BATCH=api.HParam('batch size',api.Discrete([100,300,500]))
HP_EPOCH=api.HParam('epochs',api.Discrete([500]))
METRIC_SCORE='R2'
with summary.create_file_writer(log_dir).as_default():
    api.hparams_config(hparams=[HP_LEARN,HP_NODES,HP_DROPOUT,HP_BATCH,HP_EPOCH],metrics=[api.Metric(METRIC_SCORE,display_name='R2')])

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

def train_test_model(hparams):
    #shuffle & split
    x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(X_1D,Sgh_1D,test_size=split)
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
        
    #construct neural network
    keras.backend.clear_session()
    input1=keras.Input(shape=x_train_norm[0].shape,name='well_log_data')
    NN=keras.layers.Dense(units=hparams[HP_NODES],activation=act_fun)(input1)
    NN=keras.layers.Dropout(hparams[HP_DROPOUT])(NN)
    for y in range(1,Nlayers):
        NN=keras.layers.Dense(units=hparams[HP_NODES],activation=act_fun)(NN)
        NN=keras.layers.Dropout(hparams[HP_DROPOUT])(NN)
    output1=keras.layers.Dense(units=1,activation='linear',name='gas_hydrate_saturation')(NN)
    NN=keras.Model(inputs=input1,outputs=output1)
    
    #neural network options & compilation
    NN.compile(optimizer=keras.optimizers.Adam(learning_rate=hparams[HP_LEARN]),loss=keras.losses.MeanSquaredError(),metrics=[keras.metrics.MeanAbsoluteError()])
    
    #train & test
    fitted=NN.fit(x_train_norm,numpy.array(y_train),batch_size=hparams[HP_BATCH],epochs=hparams[HP_EPOCH],verbose=0)
    evaluated=NN.evaluate(x_test_norm,numpy.array(y_test),verbose=0)
    loss=fitted.history['loss']
    
    #predict & calculate R2
    y_pred=NN.predict(x_test_norm,verbose=0).flatten()
    return sklearn.metrics.r2_score(y_test,y_pred)
    
def run(run_dir,hparams):
  with summary.create_file_writer(run_dir).as_default():
      api.hparams(hparams)
      scores=[]
      for x in range(runs_per_param):
          scores.append(train_test_model(hparams))
      scores_avg=numpy.average(scores)
      print('R2 = ',scores_avg)
      summary.scalar(METRIC_SCORE,scores_avg,step=1)
      
session=1
for learn in HP_LEARN.domain.values:
    for nodes in HP_NODES.domain.values:
        for dropout in HP_DROPOUT.domain.values:
            for batch in HP_BATCH.domain.values:
                for epochs in HP_EPOCH.domain.values:
                    hparams={
                        HP_LEARN:learn,
                        HP_NODES:nodes,
                        HP_DROPOUT:dropout,
                        HP_BATCH:batch,
                        HP_EPOCH:epochs
                        }
                    run_name="run-%d" % session
                    print('--- starting trial : %s' % run_name)
                    print({h.name: hparams[h] for h in hparams})
                    run(log_dir+run_name,hparams)
                    session+=1