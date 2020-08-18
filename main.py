import baseliner
import preprocess

if __name__ == '__main__':

    path = "RawData"

    data = preprocess.read_files(path)
    preprocess.clean_prepare_data(data)
    X_train, Y_train, X_test, Y_test = preprocess.get_train_test()

    RandomForestAccuracy = baseliner.RandomForest(X_train, Y_train, X_test, Y_test)
    SVMAccuracy = baseliner.SVM(X_train, Y_train, X_test, Y_test)
    DecisionTreeAccuracy = baseliner.DecesionTree(X_train, Y_train, X_test, Y_test)

    print("RandomForestAccuracy: ", RandomForestAccuracy)
    print("SVMAccuracy: ", SVMAccuracy)
    print("DecisionTreeAccuracy: ", DecisionTreeAccuracy)



