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
        d1 = np.load(data1[i])
        d1.resize(d1.shape[0]*d1.shape[1])
        X.append(d1)
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
    return score,score_m,score_s

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
    testnumber=[9,10,11,12]
    kernel = 'rbf'
    C = 100
    gamma = 0.01

    for testnum in testnumber:
        testmove_path =  '/home//prog/code/data/testdataset/test/move/'+str(testnum)
        teststill_path = '/home//prog/code/data/testdataset/test/still/'+str(testnum)
        model_path = '/home/zixiao/prog/code/model/svm/svm_model_'+str(kernel)+'_'+str(gamma)+'_'+str(C)+'.m'
        print(model_path)

        file = '/home//prog/code/result/SVM_'+str(kernel)+'_'+str(gamma)+'_'+str(C)+ '-test-'+str(testnum)+'_withseparateaccuracy.txt'

        test_acc,move_acc,still_acc = accuracy(model_path, testmove_path, teststill_path)


        print(f'\t Test. Acc: {test_acc * 100:.2f}%')
        print(f'\t move_acc. Acc: {move_acc * 100:.2f}%')
        print(f'\t still_acc. Acc: {still_acc * 100:.2f}%')
        with open(file, 'a') as f:

            f.write(f'test dataset {testnum}: \n'
                f'\t Test. Acc: {test_acc * 100:.2f}%\n'
                f'\t move_acc. Acc: {move_acc * 100:.2f}%\n'
                f'\t still_acc. Acc: {still_acc * 100:.2f}%\n\n'    )





