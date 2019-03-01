import scipy.io
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


dataPath='../data/og/'
lb = scipy.io.loadmat(dataPath +'VectLB19922008.mat')
depths=lb['depth']
del lb
def loadData():
    lb = scipy.io.loadmat(dataPath +'VectLB19922008.mat')
    labels = [k[0] for k in lb['labels'][0] ]
    data=pd.DataFrame(lb['Vect'],columns=labels)
    depths=lb['depth']
    dataCenter=data[np.logical_and(np.logical_and(np.logical_and(data['latitude']<=33, data['latitude']>=31 ), data['longitude']>=-65 ), data['longitude']<=-63 )]
    test=dataCenter[dataCenter['year']==2008]
    validation=dataCenter[dataCenter['year']!=2008]
    train=data[data['year']!=2008]
    return train,validation,test,depths

def split(Xvars,Yvars):
    train,val,test,depths=loadData()
    xtrains=[]
    ytrains=[]
    xvals=[]
    yvals=[]
    xtest=test.loc[:,Xvars]
    ytest=test.loc[:,Yvars]
    for i in range(1992,2008):
        xtrains.append(train.loc[train['year']!=i,Xvars])
        ytrains.append(train.loc[train['year']!=i,Yvars])
        xvals.append(val.loc[val['year']==i,Xvars])
        yvals.append(val.loc[val['year']==i,Yvars])
    return xtrains,ytrains,xvals,yvals,xtest,ytest

def applyToy(f,ytrains,yvals,ytest):
    return [f(y) for y in ytrains],[f(y) for y in yvals],f(ytest)
def evaluate(y_pred,y_true):
    return mean_squared_error(y_true,y_pred)

def evaluateFull(y_pred,y_true):
    return mean_squared_error(y_true,y_pred)

def validate(model,xtrains,ytrains,xvals,yvals):
    fold_evaluation=[]
    for xt,yt,xv,yv in zip(xtrains,ytrains,xvals,yvals):
        model.fit(xt,yt)
        yp = model.predict(xv)
        fold_evaluation.append(evaluate(yp,yv))
    return np.mean(fold_evaluation),np.std(fold_evaluation)

def plotYear(values,title="chlorophyll-a values Vertical profile for a single year"):
    plt.figure(figsize=(17,73))
    plt.xlabel("Month")
    plt.ylabel("Depth")
    plt.title(title)
    ax = plt.gca()

    ax.set_yticks(range(18))
    ax.set_yticklabels([ '%3.f' % (f) for f in depths][:17])
    im=plt.imshow(values.T,cmap='gray')

    ax.set_xticks(range(3,73,6))
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])


    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)


    plt.show()

def plotResult(values,title="chlorophyll-a values Vertical profile for a single year"):
    plt.figure(figsize=(17,73))
    ax = plt.gca()
    plt.xlabel("Time")
    plt.ylabel("Depth")
    plt.title(title)
    ax.set_yticks(range(18))
    ax.set_yticklabels([ '%3.f' % (f) for f in depths][:17])
    im=plt.imshow(values.T,cmap='gray')

    ax.set_xticks(range(3,len(values),6))
    ax.set_xticklabels([])


    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)


    plt.show()
