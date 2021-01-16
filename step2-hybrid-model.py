import pandas as pd
import numpy as np

a = pd.read_csv('./data/latent_factor.csv')
b = pd.read_csv('./data/collaborative_filtering.csv')
a = a.to_numpy()
b = b.to_numpy()
for i in range(b.shape[0]):
    # change weights depending on optimization
    b[i, 1] = 0.5 * a[i, 1] + 0.5 * b[i, 1]
b = pd.DataFrame(b)
b[[0]] = b[[0]].astype(int)
b.rename(columns={0: 'Id', 1: 'Rating'}, inplace=True)
b.to_csv('./data/hybrid.csv', index=False)
