import pandas as pd


dat_1 = pd.read_csv('750-1h-big.dat', header=None, names=['data'])
index = list(dat_1.loc[dat_1.data == 'Data Acquisition: Timed'].index)
index.pop(0)
for i in range(len(index)):
    index.append(index[i] - 1)
    index.append(index[i] + 1)
    index.append(index[i] + 2)
    index.append(index[i] + 3)
    index.append(index[i] + 4)
index.sort()
print(index)
dat_1.drop(index=index, inplace=True)
dat_1.to_csv('750-1h.txt', index=False)


