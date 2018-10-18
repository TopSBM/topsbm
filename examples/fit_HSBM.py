import os
import pylab as plt 

#from ..\hSBM import hSBMTransformer

path_data = ''

## texts
fname_data = 'corpus.txt'
filename = os.path.join(path_data,fname_data)

with open(filename,'r') as f:
    x = f.readlines()
texts = [h.split() for h in x]

## titles
fname_data = 'titles.txt'
filename = os.path.join(path_data,fname_data)

with open(filename,'r') as f:
    x = f.readlines()
titles = [h.split()[0] for h in x]

print('Reached here!')

i_doc = 0
print(titles[0])
print(texts[i_doc][:10])

# Fit the model.
model = hSBMTransformer()
