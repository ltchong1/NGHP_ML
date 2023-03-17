# -*- coding: utf-8 -*-
"""
@author: ctlek_000
"""

import os
import numpy
import sklearn.model_selection
from tensorflow import keras
directory='G:/work/NETL/hydrate/python/NGHP/morph_Kfold'
os.chdir(directory)

#hyperparameters
folds=2
act_fun='relu'
net_top=[40,40]
drop_rate=.5
learn_rate=.001
bat_size=100
eps=500
wells=[5,6,7,8,9,10]

#open & read well log data
well=[]
for x in wells:
    file=open('data'+str(x)+'.txt','r')
    lines=file.readlines()
    file.close()
    for y in range(0,len(lines)):
        splitted=lines[y].split()
        floated=[float(z) for z in splitted]
        floated.append(x)
        well.append(floated)
well=numpy.array(well)

#features
depth=well[:,0]
GR=well[:,1]
Rt=well[:,2]
density=well[:,3]
porosity=well[:,4]
Vp=well[:,5]
Sgh=well[:,6]
morph=well[:,7]
well_num=well[:,8]

#well log combination
X=numpy.transpose([GR,Rt,density,porosity,Vp])

#encode morph classes as 1-hot matrix
morph_2D=[]
for x in range(len(morph)):
    morph_2D.append([0.0,0.0,0.0])
    morph_2D[-1][int(morph[x])]=1
morph_2D=numpy.array(morph_2D)

#multiple runs with random splitting
best=[1,1,0]
loss_acc=[]
con_mat_list=[]
x=0
kf=sklearn.model_selection.KFold(n_splits=folds,shuffle=True)
for traindex,testindex in kf.split(X,morph_2D):
    x_train,x_test=X[traindex],X[testindex]
    y_train,y_test=morph_2D[traindex],morph_2D[testindex]
    
    #scale
    transformer=sklearn.preprocessing.MinMaxScaler().fit(x_train)
    x_train_norm=transformer.transform(x_train)
    x_test_norm=transformer.transform(x_test)

    #construct neural network
    keras.backend.clear_session()
    input1=keras.Input(shape=x_train_norm[0].shape,name='well_log_data')
    NN=keras.layers.Dense(units=net_top[0],activation=act_fun)(input1)
    NN=keras.layers.Dropout(drop_rate)(NN)
    for y in range(1,len(net_top)):
        NN=keras.layers.Dense(units=net_top[y],activation=act_fun)(NN)
        NN=keras.layers.Dropout(drop_rate)(NN)
    output1=keras.layers.Dense(units=3,activation='softmax',name='morphological_assignment')(NN)
    NN=keras.Model(inputs=input1,outputs=output1)
    
    #neural network options & compilation
    NN.compile(optimizer=keras.optimizers.Adam(learning_rate=learn_rate),loss=keras.losses.CategoricalCrossentropy(),metrics=[keras.metrics.CategoricalAccuracy()])
    
    #train & test
    fitted=NN.fit(x_train_norm,y_train,batch_size=bat_size,epochs=eps,verbose=0)
    evaluated=NN.evaluate(x_test_norm,y_test,verbose=0)
    loss=fitted.history['loss']
    
    #predict
    y_pred=NN.predict(x_test_norm,verbose=0)
    y_pred_1D=numpy.argmax(y_pred,axis=1)
    y_test_1D=numpy.argmax(y_test,axis=1)
    con_mat=sklearn.metrics.confusion_matrix(y_test_1D,y_pred_1D)
    
    #save & print
    loss_acc.append(evaluated)
    con_mat_list.append(con_mat)
    print('run',x+1)
    print('loss =',evaluated[0])
    print('accuracy =',evaluated[1])
    print('confusion matrix\n',con_mat)
    print(sklearn.metrics.classification_report(y_test_1D,y_pred_1D,target_names=['none','fracture','pore']))
    
    #save best run
    if evaluated[1]>best[2]:
        NN.save(directory+'/saved')
        best[0]=x+1
        best[1]=evaluated[0]
        best[2]=evaluated[1]
        best_transformer=transformer
        best_con_mat=con_mat
        
        #save best run predictions
        prediction=[]
        for y in range(len(y_pred_1D)):
            prediction.append([max(y_pred[y]),y_pred_1D[y],y_test_1D[y]])
        numpy.savetxt('loss.txt',loss,fmt='%.9f',delimiter='\t')
        numpy.savetxt('prediction.txt',prediction,fmt='%.9f',delimiter='\t')
    x+=1
numpy.savetxt('loss_accuracy.txt',loss_acc,fmt='%.9f',delimiter='\t')

#print avg & stdev confusion matrix as well as best run stats
print('confusion matrix average\n',numpy.mean(numpy.array(con_mat_list),axis=0))
print('confusion matrix standard deviation\n',numpy.std(numpy.array(con_mat_list),axis=0))
print('best run',best[0])
print('loss =',best[1])
print('accuracy =',best[2])
print('confusion matrix\n',best_con_mat)