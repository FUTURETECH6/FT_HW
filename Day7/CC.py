def LRpredict(X_train, X_test, y_train):

    # your code here begin
    # train your model on 'x_train' and 'x_test'
    # predict on 'y_train' and get 'y_pred'
    logmodel = LogisticRegression()
    logmodel.fit(X_train, y_train)
    y_pred = logmodel.predict(X_test)
    LOGCV = (cross_val_score(logmodel, X_train, y_train, cv=k_fold, n_jobs=1, scoring='accuracy').mean())
    # your code here end

    return y_pred,LOGCV

def KNNpredict(X_train, X_test, y_train):

    # your code here begin
    # train your model on 'x_train' and 'x_test'
    # predict on 'y_train' and get 'y_pred'
    knn = KNeighborsClassifier(n_neighbors=22)
    knn.fit(X_train, y_train)
    knnpred = knn.predict(X_test)
    KNNCV = (cross_val_score(knn, X_train, y_train, cv=k_fold, n_jobs=1, scoring='accuracy').mean())    # your code here end

    return knnpred,KNNCV

def SVCpredict(X_train, X_test, y_train):

    svc = SVC(kernel='sigmoid')
    svc.fit(X_train, y_train)
    svcpred = svc.predict(X_test)

    SVCCV = (cross_val_score(svc, X_train, y_train, cv=k_fold, n_jobs=1, scoring='accuracy').mean())

    return svcpred , SVCCV

def DTreepredict(X_train, X_test, y_train):

    dtree = DecisionTreeClassifier(criterion='gini')  # criterion = entopy, gini
    dtree.fit(X_train, y_train)
    dtreepred = dtree.predict(X_test)

    DTREECV = (cross_val_score(dtree, X_train, y_train, cv=k_fold, n_jobs=1, scoring='accuracy').mean())

    return dtreepred, DTREECV

def RFCpredict(X_train, X_test, y_train):

    rfc = RandomForestClassifier(n_estimators=200)  # criterion = entopy,gini
    rfc.fit(X_train, y_train)
    rfcpred = rfc.predict(X_test)

    RFCCV = (cross_val_score(rfc, X_train, y_train, cv=k_fold, n_jobs=1, scoring='accuracy').mean())

    return rfcpred, RFCCV

def Gausspredict(X_train, X_test, y_train):

    gaussiannb = GaussianNB()
    gaussiannb.fit(X_train, y_train)
    gaussiannbpred = gaussiannb.predict(X_test)
    probs = gaussiannb.predict(X_test)

    GAUSIAN = (cross_val_score(gaussiannb, X_train, y_train, cv=k_fold, n_jobs=1, scoring='accuracy').mean())

    return probs, GAUSIAN

def Gausspredict(X_train, X_test, y_train):

    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)
    xgbprd = xgb.predict(X_test)

    print(confusion_matrix(y_test, xgbprd))
    print(round(accuracy_score(y_test, xgbprd), 2) * 100)
    XGB = (cross_val_score(estimator=xgb, X=X_train, y=y_train, cv=10).mean())

    return probs, GAUSIAN

def GBKpredict(X_train, X_test, y_train):
    
    gbk = GradientBoostingClassifier()
    gbk.fit(X_train, y_train)
    gbkpred = gbk.predict(X_test)
    print(confusion_matrix(y_test, gbkpred))
    print(round(accuracy_score(y_test, gbkpred), 2) * 100)
    GBKCV = (cross_val_score(gbk, X_train, y_train, cv=k_fold, n_jobs=1, scoring='accuracy').mean())

    return gbkpred, GBKCV