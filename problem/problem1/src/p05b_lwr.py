import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    # *** START CODE HERE ***
    # Fit a LWR model
    # Get MSE value on the validation set
    # Plot validation predictions on top of training set
    # No need to save predictions
    # Plot data
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    clf = LocallyWeightedLinearRegression(tau)
    clf.fit(x_train, y_train)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = clf.predict(x_eval)
    mse = np.mean((y_pred - y_eval) ** 2)
    print(f"Validation MSE for tau={tau}: {mse}")
    plt.figure()
    plt.plot(x_train[:, 1], y_train, 'bx', label='Training Set')
    plt.plot(x_eval[:, 1], y_pred, 'ro', label='Validation Predictions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Locally Weighted Linear Regression (tau = {tau})')
    plt.legend()
    plt.savefig('output/p05b.png')
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = np.array(x)
        self.y = np.array(y)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        x = np.array(x)
        m, n = x.shape
        y = np.zeros((m, 1))
        for i in range(m):
            diff = self.x - x[i]
            dist = np.sum(np.square(diff), axis=1)
            w = np.exp(-dist/(2*self.tau*self.tau)) 
            w = np.diag(w)
            theta = np.linalg.inv(self.x.T@w@self.x)@self.x.T@w@self.y
            y[i] = theta.T@x[i]
        return y.flatten()
        # *** END CODE HERE ***
