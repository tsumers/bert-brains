import numpy as np 
from scipy.optimize import minimize 
from sklearn.linear_model import Ridge 

layers=np.random.normal(size=(12,340,768))
parcel=np.random.normal(size=(340,1))

def loss(x):
    weights=x[:12]
    alpha=x[12]
    model=Ridge(alpha=alpha,normalize=False)
    X=np.zeros((layers.shape[1:]))
    for i in range(layers.shape[0]):
        X+=weights[i]*layers[i]
    model.fit(X[:150,:],parcel[:150,:])
    model.coef_[:,-10:]=0.0
    preds=model.predict(X[150:,:])
    return np.sum((preds-parcel[150:,:])**2)

def constraints(x):
    return np.sum(x)==1.0 and np.sum(x<0)==0

cons=({'type':'eq','fun':lambda x: np.sum(x[:12])-1})
bounds=[(0.0,1.0) for _ in range(12)]
bounds.append((None,None))
x0=np.ones((13,))*(1.0/12)
x0[-1]=1.0
print(minimize(loss,x0,constraints=cons,bounds=bounds))
