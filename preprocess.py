import pandas as pd
import numpy as np
import os


def read_files(path):
    """
    This function takes raw data path as input parameter and read data files into DataFrame.
    Returns DataFrame.
    :param path:
    :return data:
    """
    try:
        files_list = os.listdir(path)
        if not files_list:
            print("Empty Directory")
            exit()
        else:
            print(files_list)
            data = pd.concat([pd.read_csv(os.path.join(path, files_list[i]), sep=",", header=None) for i in
                              range(len(files_list))],
                             ignore_index=True)
            return data

    except (IOError, OSError) as e:
        print(e)
        exit()


def clean_prepare_data(data):
    """
    This function will take DataFrame as input.
    It will clean and prepare data for Further pre-processing
    This will also create train and test data file separately and save in CleanData directory.
    :param data:
    :return None:
    """
    # remove redundant values
    data = data.replace('?', np.nan).drop_duplicates()

    # to check null values
    is_nan = data.isnull()
    row_has_nan = is_nan.any(axis=1)

    rows_with_nan = data[row_has_nan]

    # DataFrame without NaN values
    not_nan = pd.concat([data, rows_with_nan], axis=0).drop_duplicates(keep=False, ignore_index=True)

    #print(not_nan.head())
    grouped = not_nan.groupby(13)

    # separating test and train data before further processing.
    test_data = pd.concat([grouped.get_group(i).head(int((len(grouped.get_group(i)) * 40) / 100)) for i in range(5)],
                          ignore_index=True)

    train_data = pd.concat([data, test_data]).drop_duplicates(keep=False)

    # saving train test csv files
    if not os.path.exists('CleanData'):
        os.makedirs('CleanData')
        test_data.to_csv("CleanData/test.csv", sep=',', index=False)
        train_data.to_csv("CleanData/train.csv", sep=',', index=False)
    else:
        test_data.to_csv("CleanData/test.csv", sep=',', index=False)
        train_data.to_csv("CleanData/train.csv", sep=',', index=False)


def get_train_test():
    """

    :return X_train, Y_train, X_test, Y_test:
    """
    traindf = pd.read_csv("CleanData/train.csv")

    # print(traindf.head())
    grouped = traindf.groupby('13')

    cat_0 = grouped.get_group(0)
    cat_0 = cat_0.fillna(cat_0.mean())

    cat_1 = grouped.get_group(1)
    cat_1 = cat_1.fillna(cat_1.mean())

    cat_2 = grouped.get_group(2)
    cat_2 = cat_2.fillna(cat_2.mean())

    cat_3 = grouped.get_group(3)
    cat_3 = cat_3.fillna(cat_3.mean())

    cat_4 = grouped.get_group(4)
    cat_4 = cat_4.fillna(cat_4.mean())

    train = pd.concat([cat_0, cat_1, cat_2, cat_3, cat_4], axis=0).sample(frac=1).reset_index(drop=True)

    Y_train = train['13'].replace(2, 1).replace(3, 1).replace(4, 1)
    X_train = train.drop(columns=['13'])

    test = pd.read_csv("CleanData/test.csv")
    X_test = test.iloc[:, :13]
    Y_test = test['13'].replace(2, 1).replace(3, 1).replace(4, 1)

    return X_train, Y_train, X_test, Y_test
