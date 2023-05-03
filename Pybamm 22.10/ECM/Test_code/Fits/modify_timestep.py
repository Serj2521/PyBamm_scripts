
import pandas as pd
import numpy as np

# create dataframe with 60-second time steps
times = np.arange(0, 600, 60)
values = np.random.rand(len(times))
df = pd.DataFrame({'Value': values}, index=times)

# interpolate to 1-second time steps
new_index = pd.date_range(df.index.min(), df.index.max(), freq='1s')
df = df.reindex(new_index).interpolate(method='linear')

print(df.head())