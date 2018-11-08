import os
import matplotlib.pylab as plt 

from sklearn.feature_extraction.text import CountVectorizer
from topsbm import TopSBM

path_data = ''

## Load texts and vectorize
fname_data = 'corpus.txt'
filename = os.path.join(path_data,fname_data)

with open(filename,'r') as f:
    docs = f.readlines()

vec = CountVectorizer()
X = vec.fit_transform(docs)

# X is now a sparse matrix of (docs, words)

## titles
fname_data = 'titles.txt'
filename = os.path.join(path_data,fname_data)

with open(filename,'r') as f:
    x = f.readlines()
titles = [h.split()[0] for h in x]

print('Reached here!')

i_doc = 0
print(titles[0])
print(docs[i_doc][:100])

# Fit the model

model = TopSBM()
Xt = model.fit_transform(X)

model.plot_graph("model-decomposition.png")
