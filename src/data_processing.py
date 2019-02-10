import pandas as pd
import pickle
import os


def load_data_raw(train_path='../data/train', test_path='../data/test', persist=True):
    assert os.path.exists(train_path), 'missing train data'
    assert os.path.exists(train_path), 'missing test data'

    train = pd.DataFrame(columns=['Id', 'Text', 'Target'])
    test = pd.DataFrame(columns=['Id', 'Text'])

    labels = {'pos': 1, 'neg': 0}

    # read train data
    for target in labels.keys():
        for file in os.listdir('%s/%s/' % (train_path, target)):
            id = file.replace('.txt', '')
            with open('%s/%s/%s' % (train_path, target, file)) as fp:
                train = train.append({'Id': id, 'Text': fp.read(), 'Target': labels[target]}, ignore_index=True)
    # train.set_index('Id', inplace=True)

    # read test data
    for file in os.listdir(test_path):
        id = file.replace('.txt', '')
        with open('%s/%s' % (test_path, file)) as fp:
            test = test.append({'Id': id, 'Text': fp.read()}, ignore_index=True)
    # test.set_index('Id', inplace=True)

    if persist:
        pickle.dump((train, test), open('../data/dataset.dat', 'wb'))

    return train, test


def load_data_persisted():
    assert os.path.exists('../data/dataset.dat'), 'cannot find dataset'
    return pickle.load(open('../data/dataset.dat', 'rb'))


def load_data():
    if os.path.exists('../data/dataset.dat'):
        return load_data_persisted()
    else:
        return load_data_raw()
