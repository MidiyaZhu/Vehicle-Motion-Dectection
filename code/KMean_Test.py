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
        d.resize(10000)
        X.append(d)
        Y.append(0)
    data1 = get_data_path(data_path1)
    for i in tqdm(range(len(data1)), ascii=True, desc='still'):
        d1 = np.load(data1[i])
        d1.resize(10000)
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
        d.resize(10000)
        X.append(d)
        Y.append(0)
    data1 = get_data_path(stillpath)
    for i in tqdm(range(len(data1)), ascii=True, desc='still'):
        d1 = np.load(data1[i])
        d1.resize(10000)
        X.append(d1)
        Y.append(1)

    prediction=model.predict(X)
    return prediction,Y

def testmodelmove(modelpath,movepath):
    model=joblib.load(modelpath)
    data = get_data_path(movepath)
    X = []
    Y = []

    for i in tqdm(range(len(data)), ascii=True, desc='move'):
        d = np.load(data[i])
        d.resize(10000)
        X.append(d)
        Y.append(0)

    prediction=model.predict(X)
    return prediction,Y

def testmodelstill(modelpath,stillpath):
    model=joblib.load(modelpath)
    X = []
    Y = []
    data1 = get_data_path(stillpath)
    for i in tqdm(range(len(data1)), ascii=True, desc='still'):
        d1 = np.load(data1[i])
        d1.resize(10000)
        X.append(d1)
        Y.append(1)

    prediction=model.predict(X)
    return prediction,Y

if __name__=='__main__':
    testnumber = [1,2,3,4,5]
    file = '/home/zixiao/prog/code/result/KMEANs_test_withseparateaccuracy.txt'
    model_path = '/home/zixiao/prog/code/model/kmeans//minikmeans_926_u+v.m'

    with open(file, 'a') as f:

        f.write(f'training model path: {model_path}\n')
    for testnum in testnumber:
        move_path = '/home/zixiao/prog/code/data/testdataset/test/move/' + str(testnum)
        still_path = '/home/zixiao/prog/code/data/testdataset/test/still/' + str(testnum)


        y_pred, y_true = testmodel(model_path, move_path, still_path)
        y_mpred,y_m=testmodelmove(model_path,move_path)
        y_spred,y_s=testmodelstill(model_path,still_path)

        test_acc = accuracy(y_pred, y_true)
        move_acc = accuracy(y_mpred, y_m)
        still_acc = accuracy(y_spred, y_s)

        print(f'\t Test. Acc: {test_acc * 100:.2f}%')
        print(f'\t move_acc. Acc: {move_acc * 100:.2f}%')
        print(f'\t still_acc. Acc: {still_acc * 100:.2f}%')
        with open(file, 'a') as f:
            f.write(f'test dataset {testnum}: \n'
                    f'\t Test. Acc: {test_acc * 100:.2f}%\n'
                    f'\t move_acc. Acc: {move_acc * 100:.2f}%\n'
                    f'\t still_acc. Acc: {still_acc * 100:.2f}%\n\n')



