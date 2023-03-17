# -*- coding: utf-8 -*-
"""
@author: ctlek_000
"""

import os
import numpy
import sklearn.model_selection
from tensorflow import keras

directory='C:/Users/leeby/Documents/work/NETL/hydrate/python/NGHP/morph'
os.chdir(directory)

#hyperparameters
runs=100
TV_split=.2
act_fun='relu'
net_top=[125,225]
drop_rate=[.1,.2]
learn_rate=.002
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
#X=numpy.transpose([GR,Rt,density,porosity,Vp,well_num,depth])
X=numpy.transpose([GR,Rt,density,porosity,well_num,depth])

#encode morph classes as 1-hot matrix
morph_2D=[]
for x in range(len(morph)):
    morph_2D.append([0.0,0.0,0.0])
    morph_2D[-1][int(morph[x])]=1
morph_2D=numpy.array(morph_2D)

#multiple runs with random splitting
best=0
loss_acc=[]
balanced=[]
precisions=[]
recalls=[]
F_scores=[]
con_mats=[]
for x in range(runs):
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
    fitted=NN.fit(x_train_norm,y_train,batch_size=bat_size,epochs=eps,verbose=0)
    evaluated=NN.evaluate(x_test_norm,y_test,verbose=0)
    loss=fitted.history['loss']
    
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
        numpy.savetxt('depths_of_pred_none_but_exp_pore.txt',wrong_depths,fmt='%s')
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
