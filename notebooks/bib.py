import scipy.io
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.special import erfinv
from  tqdm import tqdm_notebook
dataPath='../data/og/'
#dataPath='/content/gdrive/My Drive/ocean-depth/data/og/'
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


def gausSplit(Xvars,Yvars):
    lb = scipy.io.loadmat(dataPath +'VectLB19922008.mat')
    labels = [k[0] for k in lb['labels'][0] ]
    data=pd.DataFrame(lb['Vect'],columns=labels)

    i = np.argsort( data.loc[:,Xvars], axis = 0 )
    j = np.argsort( i, axis = 0 )
    j_range = len( j ) - 1
    divider = j_range / (2-0.002)
    transformed = j / divider
    transformed = transformed - 1+0.001
    transformed = erfinv( transformed )
    data.loc[:,Xvars]=transformed


    dataCenter=data[np.logical_and(np.logical_and(np.logical_and(data['latitude']<=33, data['latitude']>=31 ), data['longitude']>=-65 ), data['longitude']<=-63 )]
    val=dataCenter[dataCenter['year']!=2008]
    test=dataCenter[dataCenter['year']==2008]
    train=data[data['year']!=2008]



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

def serialise(series,window,xvars,yvars,step=1):
    X=[]
    y=[]
    tmin=series.loc[series.index[0],'time']
    tcutMin=tmin
    tcutMax=tmin+window

    for k in range(1,len(series)-window+2,step):
        X.append(series.loc[np.logical_and(series['time']>=tcutMin,series['time']<tcutMax),xvars].values)
        y.append(series.loc[np.logical_and(series['time']>=tcutMin,series['time']<tcutMax),yvars].values)
        tcutMin=tmin+(k%73)+(k//73)*100
        tcutMax=tmin+((k+window)%73)+100*((k+window)//73)
    X=np.array(X)
    y=np.array(y)
    return X,y

def timeSplit(window,Xvars,Yvars,step=1,overlap=False):
    lb = scipy.io.loadmat(dataPath +'VectLB19922008.mat')
    labels = [k[0] for k in lb['labels'][0] ]
    data=pd.DataFrame(lb['Vect'],columns=labels)
    data['time']= data['year']*100+(data['5days']-1)
    data=data.sort_values('time')
    times=np.sort(list(set(data['time'])))
    pos=set([(k,c)for k,c in zip(data['longitude'].values,data['latitude'].values) ])

    center=[-64.02191162109375, 33.759124755859375]
    selectCenter=np.logical_and(data['longitude'].values==center[0],
                    data['latitude'].values==center[1])
    test=data.loc[np.logical_and(data['year']==2008,selectCenter),:]
    xtest,ytest=serialise(test,window,Xvars,Yvars,step)

    if overlap==False :
        xtrains=[]
        ytrains=[]
        xvals=[]
        yvals=[]
        for i in tqdm_notebook(range(1992,2008)):
            val=data.loc[np.logical_and(data['year']==i,selectCenter),:]
            a,b=serialise(val,window,Xvars,Yvars,step)
            xvals.append(a)
            yvals.append(b)

            for i,p in enumerate(pos):
                selectP=np.logical_and(data['longitude'].values==p[0],
                data['latitude'].values==p[1])

                tr1=data.loc[np.logical_and(np.logical_and(data['year']<2008,
                                                           data['year']>i),selectP),:]
                tr2=data.loc[np.logical_and(data['year']<i,selectP),:]

                a1,b1=serialise(tr1,window,Xvars,Yvars,step)

                if(i>1992):
                    a2,b2=serialise(tr2,window,Xvars,Yvars,step)
                    a=np.append(a1,a2,axis=0)
                    b=np.append(b1,b2,axis=0)
                else:
                    a=a1
                    b=b1


                if(i==0):
                    xi=a
                    yi=b
                else:
                    xi=np.append(xi,a,axis=0)
                    yi=np.append(yi,b,axis=0)

            xtrains.append(np.array(xi))
            ytrains.append(np.array(yi))

    return xtrains,ytrains,xvals,yvals,xtest,ytest

def rnnSplit(window, Xvars,Yvars,step=1,overlap=False):

    lb = scipy.io.loadmat(dataPath +'VectLB19922008.mat')
    labels = [k[0] for k in lb['labels'][0] ]
    data=pd.DataFrame(lb['Vect'],columns=labels)
    data['time']= data['year']*100+(data['5days']-1)
    data=data.sort_values('time')
    times=np.sort(list(set(data['time'])))
    pos=set([(k,c)for k,c in zip(data['longitude'].values,data['latitude'].values) ])

    center=[-64.02191162109375, 33.759124755859375]
    selectCenter=np.logical_and(data['longitude'].values==center[0],
                    data['latitude'].values==center[1])
    test=data.loc[np.logical_and(data['year']==2008,selectCenter),:]
    xtest,ytest=serialise(test,window,Xvars,Yvars,step)

    if overlap==False :
        xtrains=[]
        ytrains=[]
        xvals=[]
        yvals=[]

        for j in tqdm_notebook(range(1992,2007)):
            val=data.loc[np.logical_and(data['year']==j+1,selectCenter),:]
            a,b=serialise(val,window,Xvars,Yvars,step)
            xvals.append(a)
            yvals.append(b)

            for i,p in enumerate(pos):
                selectP=np.logical_and(data['longitude'].values==p[0],
                data['latitude'].values==p[1])
         
                tr=data.loc[np.logical_and(data['year']<=j,selectP),:]
                
                a,b=serialise(tr,window,Xvars,Yvars,step)

                if(i==0):
                    xi=a
                    yi=b
                else:
                    xi=np.append(xi,a,axis=0)
                    yi=np.append(yi,b,axis=0)

            xtrains.append(np.array(xi))
            ytrains.append(np.array(yi))

    return xtrains,ytrains,xvals,yvals,xtest,ytest

def applyToy(f,ytrains,yvals,ytest):
    return [f(y) for y in ytrains],[f(y) for y in yvals],f(ytest)
def evaluate(y_pred,y_true):
    return np.sqrt(mean_squared_error(y_true,y_pred))

def evaluateFull(y_pred,y_true):
    y_true=y_true.reshape(-1)
    y_pred=y_pred.reshape(-1)
    points=np.argsort(y_true)
    return (np.sqrt(mean_squared_error(y_true[points[:124]],y_pred[points[:124]])),
            np.sqrt(mean_squared_error(y_true[points[-124:]],y_pred[points[-124:]])))

def validate(model,xtrains,ytrains,xvals,yvals):
    fold_evaluation=[]
    for xt,yt,xv,yv in zip(xtrains,ytrains,xvals,yvals):
        model.fit(xt,yt)
        yp = model.predict(xv)
        fold_evaluation.append(evaluate(yp,yv))
    return np.mean(fold_evaluation),np.std(fold_evaluation)

def plotYear(values,title="chlorophyll-a values Vertical profile for a single year",c='gray',d=False):
    plt.figure(figsize=(17,73))
    plt.xlabel("Month")
    plt.ylabel("Depth")
    plt.title(title)
    ax = plt.gca()

    ax.set_yticks(range(18))
    ax.set_yticklabels([ '%3.f' % (f) for f in depths][:17])
    if not d:
        im=plt.imshow(values.T,cmap=c,vmin=0,vmax=1)
    else:
        im=plt.imshow(values.T,cmap=c,vmin=-0.5,vmax=0.5)

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

def climato(values,title='difference (to average climat)'):
    lb = scipy.io.loadmat(dataPath +'VectLB19922008.mat')
    labels = [k[0] for k in lb['labels'][0] ]
    data=pd.DataFrame(lb['Vect'],columns=labels)
    dataCenter=data[np.logical_and(np.logical_and(np.logical_and(data['latitude']<=33, data['latitude']>=31 ), data['longitude']>=-65 ), data['longitude']<=-63 )]
    data=dataCenter[dataCenter['year']!=2008]
    climat=data.groupby(data['5days']).mean().loc[:,['CHL '+ str(i) for i in range(2,19)]]
    plotYear(values-climat,title=title,c='bwr',d=True)
