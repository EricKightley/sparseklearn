import numpy as np

def generate_mnist_dataset(f, n_train, n_test = None):

    label_ids = ['0','3','9']
    all_indices = np.array(range(f['data'].shape[0]))
    all_labels = np.array(f['labels'])
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []


    for label in label_ids:
        n_train_label = n_train[label]
        if n_test:
            n_test_label = n_test[label]
        else:
            n_test_label = 0

        this_label_indices = all_indices[all_labels == int(label)] 
        keep_these_indices = np.random.choice(this_label_indices, 
                                 n_train_label + n_test_label,
                                 replace = False)
        keep_these_indices.sort()
        train_data.append(f['data'][list(keep_these_indices[:n_train_label])])
        train_labels.append(f['labels'][list(keep_these_indices[:n_train_label])])

        if n_test:
            test_data.append(f['data'][list(keep_these_indices[n_train_label:])])
            test_labels.append(f['labels'][list(keep_these_indices[n_train_label:])])
       
    X_train = np.vstack(train_data)
    y_train = np.hstack(train_labels)

    # shuffle the indices
    train_inds = list(range(len(y_train)))
    np.random.shuffle(train_inds)
    X_train = X_train[train_inds]
    y_train = y_train[train_inds]

    if n_test:
        X_test = np.vstack(test_data)
        y_test = np.hstack(test_labels)
        test_inds = list(range(len(y_test)))
        np.random.shuffle(test_inds)
        X_test = X_test[test_inds]
        y_test = y_test[test_inds]
        return [X_train, y_train, X_test, y_test]

    return[X_train, y_train]
