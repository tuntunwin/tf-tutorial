import numpy as np
from PIL import Image

def normalize(x): 
    min = np.min(x)
    max = np.max(x)
    print(min,max)
    return ((x - min)/(max - min) * 255).astype(int)

W=np.load('./data/W.npy')
b=np.load('./data/b.npy')
zero = np.zeros(W.shape)
nag = normalize(np.minimum(W,0))
pos = normalize(np.maximum(W,0))
print("shape", nag[:,0].reshape((28,28)).shape)
ns = nag[:,0].reshape((28,28))
print(np.expand_dims(ns, axis=2))
'''w, h = 512, 512
data = np.zeros((h, w, 3), dtype=np.uint8)
data[256, 256] = [255, 0, 0]
img = Image.fromarray(data, 'RGB')
img.save('my.png')
img.show()'''

#print(nag[nag != 255], pos[pos != 1])


    
#print(nag[nag != 0],pos[pos != 0], pos_m, nag_m, np.sum(nag), np.sum(pos))
#print(W.shape)
#print(W[W != 0])