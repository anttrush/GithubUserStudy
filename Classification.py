import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import tensorflow as tf
import random
from functools import reduce
MLDATAFILEDIR = r"D:\科研\CodeQualityAnalysis\CodeAnalysis\followerExp\mldata.csv"
CLFDATAFILEDIR = r"D:\科研\CodeQualityAnalysis\CodeAnalysis\followerExp\clfdata.csv"
FEATURESIZE = 6
LABELNUM = 5
def rebuilddata(labelNum):
    thresholds = [1.0e-07,1.0e-06,1.0e-05,1.0e-04,1.0e-03,1.0e-02]
    samples = [567575, 211564, 7678, 393, 36, 3]
    proportion = [(samples[labelNum-1]-samples[labelNum])/(samples[i-1]-samples[i]) for i in range(1,labelNum+1)]
    rebuilddatas = []
    with open(MLDATAFILEDIR,'r') as datafile:
        field = datafile.readline().strip()
        for line in datafile.readlines():
            data = list(map(float, line.strip().split(',')))
            datalabel = labelNum
            for i in range(labelNum):
                if data[-1] <= thresholds[i]:
                    datalabel = i
                    break
            data[-1] = datalabel
            if random.random() < proportion[datalabel-1]:
                rebuilddatas.append(data)
    with open(CLFDATAFILEDIR, 'w') as clfdata:
        clfdata.write(field+'\n')
        for data in rebuilddatas:
            clfdata.write(reduce(lambda x,y:x+','+y, map(str, data)) +'\n')

def getdata():
    print("getting data...")
    datadf = pd.read_csv(CLFDATAFILEDIR)
    if FEATURESIZE + 3 != datadf.shape[1]:
        print("FEATURESIZE wrong.")
        exit()
    X = datadf.ix[:,2:-1]
    Y = datadf.ix[:,'prvalue']
    print("Y(pagerank value) describe:")
    print(Y.describe())
    Y = Y.reshape(-1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42,shuffle=True)
    scaler = StandardScaler().fit(X_train)
    X_train_std = scaler.transform(X_train,copy=True)
    X_test_std = scaler.transform(X_test,copy=True)
    return X_train_std, X_test_std, Y_train, Y_test

def svcmodel(X_train_std, X_test_std, Y_train, Y_test):
    print("svc model run:")
    svr = SVC()
    svr.fit(X_train_std,Y_train)
    y_pred = svr.predict(X_test_std)
    acc = svr.score(X_test_std, Y_test)
    print("mean accuracy: %f" % acc)
    return y_pred

def dtreemodel(X_train_std, X_test_std, Y_train, Y_test):
    print("decisiontree model run:")
    DEEP, acc_max, d_acc= 2,-float('inf'),2
    feature_importances_ , y_pred_acc = [],[]
    for DEEP in range(2,10):
        dtree = DecisionTreeClassifier(max_depth=DEEP)
        dtree.fit(X_train_std,Y_train)
        y_pred = dtree.predict(X_test_std)
        acc = dtree.score(X_test_std, Y_test)
        print("DEEP: %d\tmean accuracy: %f" % (DEEP,acc))
        if acc_max <= acc:
            acc_max, d_acc, y_pred_acc, feature_importances_ = acc, DEEP, y_pred, dtree.feature_importances_
    print("model pred acc: %f when DEEP is %d" % (acc_max, d_acc))
    print("dtree feature_importances when DEEP is %d: " % d_acc, end='')
    print(feature_importances_)
    return y_pred_acc

def DNNmodel(X_train_std, X_test_std, Y_train, Y_test):
    print("DNN model run:")
    hide1size = FEATURESIZE
    rate = 0.001
    epoch = 20000
    X_train_std = np.array(X_train_std).reshape((-1,FEATURESIZE))
    X_test_std = np.array(X_test_std).reshape((-1,FEATURESIZE))
    Y_train = np.array(Y_train).reshape((-1,1))
    Y_test = np.array(Y_test).reshape((-1,1))
    enc = OneHotEncoder(sparse=False)
    Y_train = enc.fit_transform(Y_train)
    Y_test = enc.fit_transform(Y_test)

    X = tf.placeholder('float64', [None, FEATURESIZE])
    Y = tf.placeholder('float64', [None, LABELNUM])
    weight1 = tf.Variable(tf.random_normal([FEATURESIZE, hide1size], stddev=0.35, dtype='float64'))
    biase1 = tf.Variable(tf.zeros([hide1size], dtype='float64'))
    Hide1 = tf.nn.sigmoid ( tf.matmul(X, weight1) + biase1)
    weight2 = tf.Variable(tf.random_normal([hide1size, LABELNUM], stddev=0.35, dtype='float64'))
    biase2 = tf.Variable(tf.zeros([LABELNUM], dtype='float64'))
    y_1 = tf.matmul(Hide1,weight2) + biase2
    loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y,logits=y_1))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y, 1), tf.argmax(y_1, 1)), tf.float64))
    optimizer = tf.train.AdamOptimizer(rate).minimize(loss)
    init = tf.global_variables_initializer()
    y_pred = tf.argmax(y_1,1)

    sess = tf.Session()
    sess.run(init)
    for index in range(epoch):
        sess.run(optimizer, {X: X_train_std, Y: Y_train})
        if index % 1000 == 0:
            l_train = sess.run(loss, {X: X_train_std, Y: Y_train})
            l_test = sess.run(loss, {X: X_test_std, Y: Y_test})
            acc = sess.run(accuracy,{X: X_test_std, Y: Y_test})
            print("After %d steps,train_loss: %f\ttest_loss: %f\ttest_acc: %f" % (index, l_train, l_test, acc))
    w1, w2, b1, b2 = sess.run([weight1, weight2, biase1, biase2])
    y_pred_dnn = sess.run(y_pred, {X:X_test_std, Y:Y_test})
    acc = sess.run(accuracy, {X: X_test_std, Y: Y_test})
    print("mean accuracy: %f" % acc)
    print("w1,w2,b1,b2: ", end='')
    print(w1,w2,b1,b2)
    return np.add(y_pred_dnn,1)

for LABELNUM in range(2,6):
    rebuilddata(LABELNUM)
    args = getdata()
    # y_pred_svc = svcmodel(*args)
    y_pred_dtree = dtreemodel(*args)
    # y_pred_dnn = DNNmodel(*args)
#xx = list(range(1,len(y_pred_dnn)+1))
#plt.figure()
#plt.scatter(xx, args[3], color='black', linewidths=0.05, label='real')
## plt.scatter(xx, y_pred_svc.reshape(-1),'gx',linewidths=0.05, label='svc')
#plt.scatter(xx, y_pred_dtree.reshape(-1),'yx',linewidths=0.05, label='dtree')
#plt.scatter(xx, y_pred_dnn.reshape(-1),'bx',linewidths=0.05, label='dnn')
#plt.title("different model classification result")
#plt.show()

