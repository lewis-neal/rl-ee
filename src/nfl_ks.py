import csv
import scipy.stats as ss
import numpy as np

with open('data/env_stats.csv', newline='') as f:                                                                                                                                                                                                          
    reader = csv.reader(f)                                                                                                                                                                                                                                                 
    data = list(reader)
new_data = []
for row in data:
    if row[1] == 'random-play' or row[2] == 'random-play':
        continue
    new_data.append(float(row[3]))

new_data = np.array(new_data)
D, p = ss.kstest(new_data, 'norm')
print(D)
print(p)
