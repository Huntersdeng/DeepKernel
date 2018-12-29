#%%
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from deepkernel import Deepkernel
from sklearn import svm
from functools import reduce
#%%
def get_dataset(dataset, labels):
    load_data = sio.loadmat(dataset)
    X = load_data['fea']
    y = load_data['gnd']
    data = np.concatenate((X, y),axis=1)
    digit_list = []
    for label in labels:
        digit_list.append(np.array(data[np.where(data[:,-1]==label)],dtype=np.float32))
    digits = reduce(lambda x,y:np.concatenate((x,y)), digit_list[1:], digit_list[0])
    np.random.shuffle(digits)
    return digits
#%%
dataset = get_dataset('MNIST.mat', range(10))
m = dataset.shape[0]
dataset_train = dataset[0:int(0.8*m)]
dataset_eval = dataset[int(0.8*m):]
train_data = dataset_train[:,0:784]/255
train_labels = dataset_train[:,-1]
test_data = dataset_eval[:,0:784]/255
test_labels = dataset_eval[:,-1]
train_data.shape, train_labels.shape, test_data.shape, test_labels.shape
#%%
## build the deepkernel1
deepkernel = Deepkernel((28,28,1),solver='softmax')
deepkernel.add_layer('conv',{'filters':32, 'kernel_size':(2,2)},activation='relu',kernel_initializer='xaiver')
deepkernel.add_layer('maxpool',{'pool_size':(2,2),'strides':(2,2)})
deepkernel.add_layer('conv',{'filters':64, 'kernel_size':(3,3)},activation='relu',kernel_initializer='xaiver')
deepkernel.add_layer('maxpool',{'pool_size':(2,2)})
deepkernel.add_Flatten_layer()
deepkernel.add_layer('dense',{'units':1024},dropout=0.4,activation='relu')
deepkernel.add_layer('dense',{'units':128},dropout=0.4,activation='relu')
deepkernel.kernel_summary()
deepkernel.build(10, optimizer='adam')
deepkernel.model_summary()
#%% fit the deepkernel1
deepkernel.fit(train_data, train_labels, batch_size=128, epochs=1)
#%% svm with deepkernel1
m = test_data.shape[0]
clf = svm.SVC(kernel=deepkernel.kernel, cache_size=32)
clf.fit(test_data[0:int(0.5*m)], test_labels[0:int(0.5*m)])  
print(clf.score(test_data[int(0.5*m):],test_labels[int(0.5*m):]))
#%%
## svm(kernel:rbf)
#%%
dataset0_7 = get_dataset('MNIST.mat', range(8))
dataset8_9 = get_dataset('MNIST.mat', [8,9])
train_data = dataset0_7[:,0:784]/255
train_labels = dataset0_7[:,-1]
test_data = dataset8_9[:,0:784]/255
test_labels = dataset8_9[:,-1]
train_data.shape, train_labels.shape
#%%
deepkernel2 = Deepkernel((28,28,1),solver='softmax')
deepkernel2.add_layer('conv',{'filters':32, 'kernel_size':(2,2)},activation='relu',kernel_initializer='xaiver')
deepkernel2.add_layer('maxpool',{'pool_size':(2,2),'strides':(2,2)})
deepkernel2.add_layer('conv',{'filters':64, 'kernel_size':(3,3)},activation='relu',kernel_initializer='xaiver')
deepkernel2.add_layer('maxpool',{'pool_size':(2,2)})
deepkernel2.add_Flatten_layer()
deepkernel2.add_layer('dense',{'units':1024},dropout=0.4,activation='relu')
deepkernel2.add_layer('dense',{'units':128},dropout=0.4,activation='relu')
deepkernel2.kernel_summary()
deepkernel2.build(8, optimizer='adam')
deepkernel2.model_summary()
#%% fit the deepkernel2
deepkernel2.fit(train_data, train_labels, batch_size=128, epochs=1)
#%%
clf2 = svm.SVC(kernel=deepkernel2.kernel, cache_size=32)
clf2.fit(test_data[0:int(0.5*m)], test_labels[0:int(0.5*m)])  
print(clf2.score(test_data[int(0.5*m):],test_labels[int(0.5*m):]))
