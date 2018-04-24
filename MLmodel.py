import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
import tensorflow as tf
MLDATAFILEDIR = r"D:\科研\CodeQualityAnalysis\CodeAnalysis\followerExp\mldata.csv"
FEATURESIZE = 6
def getdata():
    # print("pagerank:")
    # pr.pagerank()
    # userDict = { userId : User object }
    # prSortedList = [ [uid, prvalue] ]
    print("getting data...")
    datadf = pd.read_csv(MLDATAFILEDIR)
    if FEATURESIZE + 3 != datadf.shape[1]:
        print("FEATURESIZE wrong.")
        exit()
    X = datadf.ix[:,2:-1]
    Y = datadf.ix[:,'prvalue']
    print("Y(pagerank value) describe:")
    print(Y.describe())
    # Y = Y.reshape(-1)
    # Y scaling
    Y = scale(Y.reshape(-1))
    print("Y scaled:\nmean: %f, std: %f" % (Y.mean(), Y.std()))
    # print(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42,shuffle=True)
    scaler = StandardScaler().fit(X_train)
    X_train_std = scaler.transform(X_train,copy=True)
    X_test_std = scaler.transform(X_test,copy=True)
    return X_train_std, X_test_std, Y_train, Y_test

def linearmodel(X_train_std, X_test_std, Y_train, Y_test):
    print("linear model run:")
    linreg = LinearRegression()
    linreg.fit(X_train_std, Y_train)
    y_pred = linreg.predict(X_test_std)
    MSE = metrics.mean_squared_error(Y_test,y_pred)
    RMSE = MSE**0.5
    R2 = metrics.r2_score(Y_test,y_pred)
    print("model pred RMSE: %f" % RMSE)
    print("model pred R2: %f" % R2)
    print("linear coef: ", end='')
    print(linreg.coef_)
    return y_pred

def svrmodel(X_train_std, X_test_std, Y_train, Y_test):
    print("svr model run:")
    svr = SVR()
    svr.fit(X_train_std,Y_train)
    y_pred = svr.predict(X_test_std)
    MSE = metrics.mean_squared_error(Y_test,y_pred)
    RMSE = MSE**0.5
    R2 = metrics.r2_score(Y_test,y_pred)
    print("model pred RMSE: %f" % RMSE)
    print("model pred R2: %f" % R2)
    print(svr.score(X_test_std,Y_test))
    return y_pred

def dtreemodel(X_train_std, X_test_std, Y_train, Y_test):
    print("decisiontree model run:")
    DEEP, rmse_min, d_rmse, r2_max, d_r2 = 2,float('inf'),2,0,2
    feature_importances_ , y_pred_r2 = [],[]
    for DEEP in range(2,20):
        dtree = DecisionTreeRegressor()
        dtree.fit(X_train_std,Y_train)
        y_pred = dtree.predict(X_test_std)
        MSE = metrics.mean_squared_error(Y_test,y_pred)
        RMSE = MSE**0.5
        R2 = metrics.r2_score(Y_test, y_pred)
        print("DEEP: %d\tRMSE: %f\tR2: %f" %(DEEP, RMSE, R2))
        if rmse_min > RMSE:
            rmse_min, d_rmse = RMSE,DEEP
        if r2_max < R2:
            r2_max, d_r2, feature_importances_, y_pred_r2 = R2, DEEP, dtree.feature_importances_, y_pred
    print("model pred RMSE: %f when DEEP is %d" % (rmse_min, d_rmse))
    print("model pred R2: %f when DEEP is %d" % (r2_max, d_r2))
    print("dtree feature_importances when DEEP is %d: " % d_r2, end='')
    print(feature_importances_)
    return y_pred_r2

def DNNmodel(X_train_std, X_test_std, Y_train, Y_test):
    print("DNN model run:")
    hide1size = FEATURESIZE
    rate = 0.001
    epoch = 5000
    X_train_std = np.array(X_train_std).reshape((-1,FEATURESIZE))
    X_test_std = np.array(X_test_std).reshape((-1,FEATURESIZE))
    Y_train = np.array(Y_train).reshape((-1,1))
    Y_test = np.array(Y_test).reshape((-1,1))

    X = tf.placeholder('float64', [None, FEATURESIZE])
    Y = tf.placeholder('float64', [None, 1])
    weight1 = tf.Variable(tf.random_normal([FEATURESIZE, hide1size], stddev=0.35, dtype='float64'))
    biase1 = tf.Variable(tf.zeros([hide1size], dtype='float64'))
    Hide1 = tf.matmul(X, weight1) + biase1
    weight2 = tf.Variable(tf.random_normal([hide1size, 1], stddev=0.35, dtype='float64'))
    biase2 = tf.Variable(tf.zeros([1], dtype='float64'))
    y_pred = tf.matmul(Hide1,weight2) + biase2
    loss = tf.reduce_mean(tf.pow(y_pred-Y, 2))
    optimizer = tf.train.GradientDescentOptimizer(rate).minimize(loss)
    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)
    for index in range(epoch):
        sess.run(optimizer, {X: X_train_std, Y: Y_train})
        if index % 500 == 0:
            l_train = sess.run(loss, {X: X_train_std, Y: Y_train})
            l_test = sess.run(loss, {X: X_test_std, Y: Y_test})
            print("After %d steps,train_loss: %f\ttest_loss: %f" % (index, l_train, l_test))
    w1, w2, b1, b2 = sess.run([weight1, weight2, biase1, biase2])
    y_pred_dnn = sess.run(y_pred, {X:X_test_std, Y:Y_test})
    MSE = sess.run(loss, {X:X_test_std, Y:Y_test})
    RMSE = MSE**0.5
    R2 = metrics.r2_score(Y_test.reshape(-1), y_pred_dnn.reshape(-1))
    print("model pred RMSE: %f" % RMSE)
    print("model pred R2: %f" % R2)
    print("w1,w2,b1,b2: ", end='')
    print(w1,w2,b1,b2)
    return y_pred_dnn

args = getdata()
y_pred_linear = linearmodel(*args)
y_pred_dtree = dtreemodel(*args)
y_pred_dnn = DNNmodel(*args)
y_pred_svr = svrmodel(*args)
xx = list(range(1,len(y_pred_linear)+1))
plt.figure()
plt.scatter(xx, args[3], color='black', linewidths=0.05, label='real')
plt.scatter(xx, y_pred_linear.reshape(-1),'rx',linewidths=0.05, label='linear')
plt.scatter(xx, y_pred_dtree.reshape(-1),'yx',linewidths=0.05, label='dtree')
plt.scatter(xx, y_pred_dnn.reshape(-1),'bx',linewidths=0.05, label='dnn')
plt.scatter(xx, y_pred_svr.reshape(-1),'gx',linewidths=0.05, label='svr')
plt.title("different model regression result")
plt.show()

