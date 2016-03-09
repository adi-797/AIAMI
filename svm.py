from sklearn import svm


def call_sklearn(Ce=1.8333, Gamma=0.1366, cache=1000):
    clf = svm.SVC(C=Ce, gamma=Gamma, cache_size=cache)
    return clf


def trainSVM(features, classes, call=False, clf=None):
    if call:
        clf = call_sklearn()

    clf.fit(features, classes)

    return clf
