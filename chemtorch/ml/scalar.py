from sklearn.preprocessing import MinMaxScaler
import numpy as np

def get_scalar(x):
    scalar = MinMaxScaler(feature_range=(0,1))
    scalar.fit(x)

    b = scalar.transform(np.zeros((1, x.shape[-1])))
    a = scalar.transform(np.ones((1, x.shape[-1]))) - b

    return a, b

