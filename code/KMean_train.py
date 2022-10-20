from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import  classification_report
import joblib
import numpy as np
import os
from tqdm import tqdm
import torch


def get_data_path(boxpath):
    box=os.listdir(boxpath)

    boxs=[]
    for count in range(len(box)):
        im_name=box[count]
        im_path=os.path.join(boxpath,im_name)
        boxs.append(im_path)

    return boxs

def accuracy(predict,labels):
    if len(predict) != len(labels):
        return -1
    else:
        predict_tensor = torch.tensor(predict)
        labels_tensor = torch.tensor(labels)
        correct = predict_tensor.eq(labels_tensor)
    return correct.sum()/len(predict)

def train_model(model,model_path,data_path,data_path1,):

    data = get_data_path(data_path)
    X = []
    Y = []

    for i in tqdm(range(len(data)), ascii=True, desc='move'):
        d = np.load(data[i])
        d.resize(d1.shape[0]*d1.shape[1])
        X.append(d)
        Y.append(0)
    data1 = get_data_path(data_path1)
    for i in tqdm(range(len(data1)), ascii=True, desc='still'):
        d1 = np.load(data1[i])
        d1.resize(d1.shape[0]*d1.shape[1])
        X.append(d1)
        Y.append(1)

    x, y = shuf(X, Y)
    model.fit(x,y)
    joblib.dump(model,model_path)

def shuf(X,Y):
    XX,YY=[],[]
    list = [i for i in range(len(X))]
    np.random.shuffle(list)
    for it in list:
        XX.append(X[it])
        YY.append(Y[it])
    return XX,YY

def testmodel(modelpath,movepath,stillpath):
    model=joblib.load(modelpath)
    data = get_data_path(movepath)
    X = []
    Y = []

    for i in tqdm(range(len(data)), ascii=True, desc='move'):
        d = np.load(data[i])
        d.resize(d1.shape[0]*d1.shape[1])
        X.append(d)
        Y.append(0)
    data1 = get_data_path(stillpath)
    for i in tqdm(range(len(data1)), ascii=True, desc='still'):
        d1 = np.load(data1[i])
        d1.resize(d1.shape[0]*d1.shape[1])
        X.append(d1)
        Y.append(1)

    prediction=model.predict(X)
    return prediction,Y

if __name__=='__main__':

    tuned_parameters=[{'init':['k-means++'], 'n_clusters':[2], 'batch_size':[32],'n_init':[10], 'max_no_improvement':[926*2], 'verbose':[0],'compute_labels':[True]}]
    scores=['precision','recall']


    move_path =  '/data/npy/train/mvoe/'
    still_path = '/data/npy/train/still/'

    model_path = '/home//prog/code/model/kmeans//kmeans.m'
    clf = GridSearchCV(MiniBatchKMeans(), tuned_parameters, cv=5, scoring='f1')
    if os.path.exists(model_path) == 0:
        train_model(clf, model_path, move_path, still_path)
        print('training done!')

    else:
        raise AssertionError(
            "File exists, [{f}] given".format(f=model_path))
    y_pred, y_true = testmodel(model_path, move_path, still_path)
    train_acc = accuracy(y_pred, y_true)
   


    print(f'\tTrain Acc: {train_acc * 100:.2f}%')

    file = '/home/zixiao/prog/code/result/KMEANs.txt'
    with open(file, 'a') as f:
        f.write(f'training model path: {model_path}\n')
        f.write(f'\tTrain Acc: {train_acc * 100:.2f}%\n\n'
                )
   
