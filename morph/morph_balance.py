# -*- coding: utf-8 -*-
"""
@author: ctlek_000
"""

import os
import numpy
import random
import sklearn.model_selection
from tensorflow import keras

random.seed(20220602)
directory='C:/Users/leeby/Documents/work/NETL/hydrate/python/NGHP/morph'
os.chdir(directory)

#hyperparameters
runs=100
TV_split=.2
act_fun='relu'
net_top=[50,100]
drop_rate=[.1,.1]
learn_rate=.0025
bat_size=100
eps=500
wells=[5,6,7,8,9,10]

#open & read well log data
well=[]
for x in wells:
    file=open('data'+str(x)+'.txt','r')
    lines=file.readlines()
    file.close()
    for y in range(len(lines)):
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

#well log combination + well number & depth
X=numpy.transpose([GR,Rt,density,porosity,Vp,well_num,depth])
#X=numpy.transpose([GR,Rt,density,porosity,well_num,depth])

#encode morph classes as 1-hot matrix
morph_2D=[]
for x in range(len(morph)):
    morph_2D.append([0.0,0.0,0.0])
    morph_2D[-1][int(morph[x])]=1
morph_2D=numpy.array(morph_2D)

second=max([numpy.count_nonzero(morph==1),numpy.count_nonzero(morph==2)])

#multiple runs with random splitting
best=0
loss_acc=[]
balanced=[]
precisions=[]
recalls=[]
F_scores=[]
con_mats=[]
all_well_con_mats=[]
all_well_balanced=[]
all_well_precisions=[]
all_well_recalls=[]
all_well_Fscores=[]
for x in range(runs):
    #balance 'none' class by limiting to 2nd highest amount
    morph2=morph
    morph2_2D=morph_2D
    X2=X
    while numpy.count_nonzero(morph2==0)>second:
        i=random.randrange(len(morph2))
        if morph2[i]==0:
            morph2=numpy.delete(morph2,i)
            morph2_2D=numpy.delete(morph2_2D,i,axis=0)
            X2=numpy.delete(X2,i,axis=0)
    
    #shuffle & split then save & delete well number & depth of test data
    x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(X,morph_2D,test_size=TV_split)
    test_num=x_test[:,-2]
    test_depth=x_test[:,-1]
    x_train=numpy.delete(x_train,[-2,-1],axis=1)
    x_test=numpy.delete(x_test,[-2,-1],axis=1)
    
    #scale
    transformer=sklearn.preprocessing.MinMaxScaler().fit(x_train)
    x_train_norm=transformer.transform(x_train)
    x_test_norm=transformer.transform(x_test)

    #construct neural network
    keras.backend.clear_session()
    input1=keras.Input(shape=x_train_norm[0].shape,name='well_log_data')
    NN=keras.layers.Dense(units=net_top[0],activation=act_fun)(input1)
    NN=keras.layers.Dropout(drop_rate[0])(NN)
    for y in range(1,len(net_top)):
        NN=keras.layers.Dense(units=net_top[y],activation=act_fun)(NN)
        NN=keras.layers.Dropout(drop_rate[y])(NN)
    output1=keras.layers.Dense(units=3,activation='softmax',name='morphological_assignment')(NN)
    NN=keras.Model(inputs=input1,outputs=output1)
    
    #neural network options & compilation
    NN.compile(optimizer=keras.optimizers.Adam(learning_rate=learn_rate),loss=keras.losses.CategoricalCrossentropy(),metrics=[keras.metrics.CategoricalAccuracy()])
    
    #train & test
    validata=(x_test_norm,y_test)
    fitted=NN.fit(x_train_norm,y_train,batch_size=bat_size,epochs=eps,verbose=0,validation_data=validata)
    evaluated=NN.evaluate(x_test_norm,y_test,verbose=0)
    loss=fitted.history['loss']
    val_acc=fitted.history['val_categorical_accuracy']
    
    #predict
    y_pred=NN.predict(x_test_norm,verbose=0)
    y_pred_1D=numpy.argmax(y_pred,axis=1)
    y_test_1D=numpy.argmax(y_test,axis=1)
    
    #save & print
    loss_acc.append(evaluated)
    balanced.append(sklearn.metrics.balanced_accuracy_score(y_test_1D,y_pred_1D))
    precisions.append(sklearn.metrics.precision_score(y_test_1D,y_pred_1D,average=None))
    recalls.append(sklearn.metrics.recall_score(y_test_1D,y_pred_1D,average=None))
    F_scores.append(sklearn.metrics.f1_score(y_test_1D,y_pred_1D,average=None))
    con_mats.append(sklearn.metrics.confusion_matrix(y_test_1D,y_pred_1D))
    print('run',x+1)
    print('loss =',evaluated[0])
    print('accuracy =',evaluated[1])
    print('balanced accuracy =',balanced[-1])
    print('confusion matrix\n',con_mats[-1])
    print(sklearn.metrics.classification_report(y_test_1D,y_pred_1D,target_names=['none','fracture','pore']))
    
    #save best run
    if balanced[-1]>balanced[best]:
        NN.save(directory+'/saved')
        best=x
        best_transformer=transformer
        
        #save best run predictions & breakdown wrong pore recall by well & depths
        prediction=[]
        wrong_depths=[[],[],[],[],[],[]]
        for y in range(len(y_pred_1D)):
            prediction.append([max(y_pred[y]),y_pred_1D[y],y_test_1D[y],test_num[y],test_depth[y]])
            if y_pred_1D[y]==0 and y_test_1D[y]==2:
                wrong_depths[int(test_num[y])-5].append(test_depth[y])
        numpy.savetxt('loss.txt',loss,fmt='%.9f',delimiter='\t')
        numpy.savetxt('prediction.txt',prediction,fmt='%.9f',delimiter='\t')
        numpy.savetxt('val_acc.txt',val_acc,fmt='%.9f',delimiter='\t')
        numpy.savetxt('depths_of_pred_none_but_exp_pore.txt',wrong_depths,fmt='%s')
        
    #metrics for each well
    well_con_mats=[]
    well_precisions=[]
    well_recalls=[]
    well_Fscores=[]
    well_balanced=[]
    three_classes=True
    for y in wells:
        wellX=X[numpy.where(X[:,5]==y)]
        wellY=morph_2D[numpy.where(X[:,5]==y)]
        if y==10:
            #wellXY=numpy.concatenate((wellX,wellY),axis=1)
            wellX=wellX[wellY[:,-1]==0]
            wellY=wellY[wellY[:,-1]==0]
        wellX=numpy.delete(wellX,[-2,-1],axis=1)
        wellX_norm=transformer.transform(wellX)
        wellY_pred=NN.predict(wellX_norm,verbose=0)
        wellY_pred_1D=numpy.argmax(wellY_pred,axis=1)
        wellY_1D=numpy.argmax(wellY,axis=1)
        well_con_mats.append(sklearn.metrics.confusion_matrix(wellY_1D,wellY_pred_1D))
        well_balanced.append(sklearn.metrics.balanced_accuracy_score(wellY_1D,wellY_pred_1D))
        well_precisions.append(sklearn.metrics.precision_score(wellY_1D,wellY_pred_1D,average=None,zero_division=0))
        well_recalls.append(sklearn.metrics.recall_score(wellY_1D,wellY_pred_1D,average=None,zero_division=0))
        F=sklearn.metrics.f1_score(wellY_1D,wellY_pred_1D,average=None,zero_division=0)
        well_Fscores.append(F)
        if F.size != 3:
            three_classes=False
    if three_classes:
        all_well_con_mats.append(well_con_mats)
        all_well_balanced.append(well_balanced)
        all_well_precisions.append(well_precisions)
        all_well_recalls.append(well_recalls)
        all_well_Fscores.append(well_Fscores)
avg_well_con_mats=numpy.mean(all_well_con_mats,axis=0)
std_well_con_mats=numpy.std(all_well_con_mats,axis=0)
avg_well_balanced=numpy.mean(all_well_balanced,axis=0)
std_well_balanced=numpy.std(all_well_balanced,axis=0)
avg_well_precisions=numpy.mean(all_well_precisions,axis=0)
std_well_precisions=numpy.std(all_well_precisions,axis=0)
avg_well_recalls=numpy.mean(all_well_recalls,axis=0)
std_well_recalls=numpy.std(all_well_recalls,axis=0)
avg_well_Fscores=numpy.mean(all_well_Fscores,axis=0)
std_well_Fscores=numpy.std(all_well_Fscores,axis=0)

numpy.savetxt('loss_accuracy.txt',loss_acc,fmt='%.9f',delimiter='\t')
numpy.savetxt('balanced.txt',balanced,fmt='%.9f',delimiter='\t')
numpy.savetxt('precisions.txt',precisions,fmt='%.9f',delimiter='\t')
numpy.savetxt('recalls.txt',recalls,fmt='%.9f',delimiter='\t')
numpy.savetxt('F_scores.txt',F_scores,fmt='%.9f',delimiter='\t')

#print avg & stdev confusion matrix as well as best run stats
print('confusion matrix average\n',numpy.mean(numpy.array(con_mats),axis=0))
print('confusion matrix standard deviation\n',numpy.std(numpy.array(con_mats),axis=0))
print('best run',best+1)
print('loss =',loss_acc[best][0])
print('accuracy =',loss_acc[best][1])
print('balanced accuracy =',balanced[best])
print('confusion matrix\n',con_mats[best])

#test best on all
X_norm=best_transformer.transform(numpy.delete(X,[-2,-1],axis=1))
best_NN=keras.models.load_model(directory+'/saved')
pred=numpy.argmax(best_NN.predict(X_norm),axis=1)
print('applying best model to all data')
print(sklearn.metrics.confusion_matrix(morph,pred))
print(sklearn.metrics.classification_report(morph,pred,target_names=['none','fracture','pore']))
