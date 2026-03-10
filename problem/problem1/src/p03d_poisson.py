import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set

    # The line below is the original one from Stanford. It does not include the intercept, but this should be added.
    # x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    clf = PoissonRegression(step_size=lr)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_eval)   
    plt.figure()
    plt.plot(y_eval, y_pred, 'bx')
    plt.xlabel('true counts')
    plt.ylabel('predict counts')
    plt.savefig('output/p03d.png')

    np.savetxt(pred_path, y_pred)

    mask = (y_pred >= 0.99 * y_eval) & (y_pred <= 1.01 * y_eval)
    N = np.sum(mask)
    M = len(y_eval)
    print('正确率为%d/%d' % (N, M))

    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        x = np.array(x)
        m, n = x.shape
        y = np.array(y).reshape(m, 1)
        self.theta = np.zeros((n, 1))
        while True:
            last_theta = self.theta
            self.theta = self.theta + self.step_size*x.T@(y - np.exp(x@self.theta))/m
            if np.sum(np.abs(self.theta - last_theta)) < self.eps: break
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        x = np.array(x)
        y = np.exp(x@self.theta)
        return y.flatten()
        # *** END CODE HERE ***
