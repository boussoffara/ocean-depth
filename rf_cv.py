from bib import *
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

xtrains,ytrains,xvals,yvals,xtest,ytest=split(('CHL 1','THERM 1','SSH','SR','WS'),
                                               ['CHL '+ str(i) for i in range(2,19)])
ytrains,yvals,ytest=applyToy(lambda x:np.log(x)/np.log(10),ytrains,yvals,ytest)


#### Cross Validation 
from sklearn.multioutput import MultiOutputRegressor

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 1, stop = 1000, num = 50)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
grid = {'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap}

rf = RandomForestRegressor()

def validate(params,xtrains,ytrains,xvals,yvals):
    fold_evaluation=[]
    first= True
    for xt,yt,xv,yv in zip(xtrains,ytrains,xvals,yvals):
        

    
        model=RandomForestRegressor(n_estimators=params['n_estimators'], max_depth=params['max_depth'], 
                                    min_samples_split=params['min_samples_split'], 
                                    min_samples_leaf=params['min_samples_leaf'], 
                                    max_features=params['max_features'], bootstrap=params['bootstrap'])
        multioutputregressor = MultiOutputRegressor(model,n_jobs=-1).fit(xt.values,yt.values)
        yp=multioutputregressor.predict(xv)
        #check error in original space
            
        fold_evaluation.append(evaluate(yp,yv))
        if first:
            print(evaluate(yp,yv))
            first=False
        
    return np.mean(fold_evaluation),np.std(fold_evaluation)

i=0
while True:
    params={}
    for key in grid:
        params[key]=grid[key][i%len(grid[key])]
#         print(i%len(parameters_for_testing[key]))
        i+=13
    print(params)
    error,std=validate(params,xtrains,ytrains,xvals,yvals)
    print('\t error: %.5f +- %.5f' % (error,std))
    






