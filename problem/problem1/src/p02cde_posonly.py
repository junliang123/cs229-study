import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)
    clf_c = LogisticRegression()
    clf_c.fit(x_train, t_train)
    util.plot(x_test, t_test, clf_c.theta, 'output/p02c.png')

    t_pred_c = clf_c.predict(x_test)
    t_pred_c = t_pred_c > 0.5
    np.savetxt(pred_path_c, t_pred_c, fmt='%.1lf')

    N = np.sum(t_test == t_pred_c)
    M = len(t_pred_c)
    print('成功率为%d/%d' % (N, M))

    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, label_col='y', add_intercept=True)
    clf_d = LogisticRegression()
    clf_d.fit(x_train, y_train)
    util.plot(x_test, t_test, clf_d.theta, 'output/p02d.png')

    y_pred_d = clf_d.predict(x_test)
    y_pred_d = y_pred_d > 0.5
    np.savetxt(pred_path_d, y_pred_d, fmt='%.1lf')

    N = np.sum(t_test == y_pred_d)
    M = len(y_pred_d)
    print('成功率为%d/%d' % (N, M))

    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    x_valid, y_valid = util.load_dataset(valid_path, label_col='y', add_intercept=True)
    x_valid = x_valid[y_valid == 1]
    y_valid = y_valid[y_valid == 1]
    y_pred_e = clf_d.predict(x_valid)
    alpha = np.sum(y_pred_e)/len(y_valid)
    correction = 1 + np.log(2 / alpha - 1) / clf_d.theta[0]
    util.plot(x_test, t_test, clf_d.theta, 'output/p02e.png', correction)
    
    y_pred_d = clf_d.predict(x_test)
    y_pred_d = y_pred_d/alpha
    y_pred_d = y_pred_d > 0.5
    np.savetxt(pred_path_e, y_pred_d, fmt='%.1lf')

    N = np.sum(t_test == y_pred_d)
    M = len(y_pred_d)
    print('成功率为%d/%d' % (N, M))
    # *** END CODER HERE
