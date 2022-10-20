from  sklearn import svm
import joblib
import numpy as np
import os
from tqdm import tqdm


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
        d = np.load(data1[i])
        d.resize(d1.shape[0]*d1.shape[1])
        X.append(d)
        Y.append(1)

    x, y = shuf(X, Y)
    model.fit(x,y)
    joblib.dump(model,model_path)

def pred_model(model_path,X):
    model=joblib.load(model_path)
    predict=model.predict(X)

    return predict

def pred_pro(model_path,X):
    model=joblib.load(model_path)
    probability=model.predict_proba(X)

    return probability

def accuracy(model_path,data_path=None,data_path1=None):
    model=joblib.load(model_path)
    data = get_data_path(data_path)
    X = []
    Y = []
    sx=[]
    sy=[]
    print(model_path)
    for i in tqdm(range(len(data)), ascii=True, desc='move'):
        d = np.load(data[i])
        d.resize(d1.shape[0]*d1.shape[1])
        X.append(d)
        Y.append(0)
    score_m=model.score(X,Y)
    print('move accuracy: ',score_m)
    data1 = get_data_path(data_path1)
    for i in tqdm(range(len(data1)), ascii=True, desc='still'):
        d1 = np.load(data1[i])
        d1.resize(d1.shape[0]*d1.shape[1])
        X.append(d1)
        sx.append(d1)
        Y.append(1)
        sy.append(1)
    score_s=model.score(sx,sy)
    print('still accuracy: ', score_s)
    score=model.score(X,Y)
    print('total accuracy: ',score)
    return score

def get_data_path(boxpath):
    box=os.listdir(boxpath)

    boxs=[]
    for count in range(len(box)):
        im_name=box[count]
        im_path=os.path.join(boxpath,im_name)
        boxs.append(im_path)

    return boxs

def shuf(X,Y):
    XX,YY=[],[]
    list = [i for i in range(len(X))]
    np.random.shuffle(list)
    for it in list:
        XX.append(X[it])
        YY.append(Y[it])
    return XX,YY



if __name__=='__main__':
    kernel='rbf'
    C=100
    gamma=0.01
    model = svm.SVC(kernel=kernel, C=C, gamma=gamma, probability=True)

    move_path =  '/data/npy/train/move/'
    still_path = '/data/npy/train/still/'
    model_path = '/home//prog/code/model/svm/svm_model_'+str(kernel)+'_'+str(gamma)+'_'+str(C)+'.m'
    print(model_path)

    file = '/home/prog/code/log/log_SVM_'+str(kernel)+'_'+str(gamma)+'_'+str(C)+ '.txt'

    if os.path.exists(model_path) == 0:
        train_model(model,model_path, move_path, still_path)
        print('training done!')

    else:
        raise AssertionError(
            "File exists, [{f}] given".format(f=model_path))
    train_acc = accuracy(model_path, move_path, still_path)

  
    print(f'\tTrain Acc: {train_acc * 100:.2f}%')

    with open(file, 'a') as f:

        f.write(f'\tTrain Acc: {train_acc * 100:.2f}%\n'
              )
