import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d

input_path = '../data/FC2_With_Ripples_Excel.csv'
# FC1_Without_Ripples_Excel
# FC2_With_Ripples_Excel.csv
df = pd.read_csv(input_path, sep=',')

time_col = 'Time (h)'
df[time_col] = df[time_col].astype(float)

max_time = df[time_col].max()
sampled_times = np.arange(0, max_time + 0.5, 0.5)

sampled_df = pd.DataFrame({time_col: sampled_times})
sampled_df = pd.merge_asof(sampled_df.sort_values(time_col),
                          df.sort_values(time_col),
                          on=time_col,
                          direction='nearest')

def apply_gaussian_filter(series, sigma=2):
    return gaussian_filter1d(series, sigma=sigma, mode='nearest')

start_col = 1
end_col = 6
filter_columns = sampled_df.columns[start_col:end_col]

for col in filter_columns:
    sampled_df[col+'_filtered'] = apply_gaussian_filter(sampled_df[col], sigma=2)

output_path = 'output/FC2.csv'
# FC1
# FC2
sampled_df.to_csv(output_path, index=False)