from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import tree

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

def RandomForest(X_train, Y_train, X_test, Y_test):
    """

    :param X_train:
    :param Y_train:
    :param X_test:
    :param Y_test:
    :return acc:
    """
    Y_train = Y_train.to_numpy().reshape(-1, 1)
    Y_train = OneHotEncoder().fit([[0], [1]]).transform(Y_train).toarray()

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)

    clf = RandomForestClassifier(max_depth=5, random_state=0)
    clf.fit(X_train, Y_train)

    X_test = scaler.transform(X_test)
    Y_test = Y_test.to_numpy().reshape(-1, 1)
    Y_test = OneHotEncoder().fit([[0], [1]]).transform(Y_test).toarray()

    acc = clf.score(X_test, Y_test)
    #print("Acuracy: ", acc)
    return acc


def SVM(X_train, Y_train, X_test, Y_test):
    """
    Takes parameters specified in param and applies SVM classifier on the train test set.
    Returns Accuracy
    :param X_train:
    :param Y_train:
    :param X_test:
    :param Y_test:
    :return acc:
    """

    scaler = preprocessing.StandardScaler().fit(X_train)

    X_train = scaler.transform(X_train)
    Y_train = Y_train.to_numpy()

    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X_train, Y_train)

    X_test = scaler.transform(X_test)
    Y_test = Y_test.to_numpy()

    acc = clf.score(X_test, Y_test)

    #print("Acuracy: ", acc)
    return acc


def DecesionTree(X_train, Y_train, X_test, Y_test):

    Y_train = Y_train.to_numpy().reshape(-1, 1)
    Y_train = OneHotEncoder().fit([[0], [1]]).transform(Y_train).toarray()

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)

    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, Y_train)

    X_test = scaler.transform(X_test)
    Y_test = Y_test.to_numpy().reshape(-1, 1)
    Y_test = OneHotEncoder().fit([[0], [1]]).transform(Y_test).toarray()

    acc = clf.score(X_test, Y_test)
    #print("Acuracy: ", acc)
    return acc




