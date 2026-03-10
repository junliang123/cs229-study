import numpy as np
import util

from linear_model import LinearModel

import matplotlib.pyplot as plt

def plot_decision_boundary(x, y, theta, save_path="output.png"):
    """
    绘制数据散点图和逻辑回归的决策边界
    注意：这里的 x 包含了截距项列 (x0=1)
    """
    plt.figure(figsize=(8, 6))
    
    # 将 y 展平以匹配索引
    y = y.flatten()
    
    # 1. 绘制散点图
    # x[:, 1] 是第一个特征，x[:, 2] 是第二个特征
    plt.plot(x[y == 0, 1], x[y == 0, 2], 'bx', label='y=0')
    plt.plot(x[y == 1, 1], x[y == 1, 2], 'go', label='y=1')
    
    # 2. 准备横坐标 x1 的范围
    x1_min, x1_max = x[:, 1].min(), x[:, 1].max()
    x1_vals = np.array([x1_min - 0.5, x1_max + 0.5])
    
    # 3. 计算对应的纵坐标 x2
    # 直线方程: theta[0]*1 + theta[1]*x1 + theta[2]*x2 = 0
    # 推导出: x2 = -(theta[0] + theta[1]*x1) / theta[2]
    theta = theta.flatten()
    x2_vals = -(theta[0] + theta[1] * x1_vals) / theta[2]
    
    # 4. 绘制决策边界
    plt.plot(x1_vals, x2_vals, 'r--', linewidth=2, label='Logistic Regression')
    
    # 设置图表属性
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Dataset 1: Logistic Regression')
    plt.legend()
    
    # 保存并显示图片
    plt.savefig(save_path)

def main(train_path, eval_path, pred_path, plot_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    m, n = x_train.shape
    clf = LogisticRegression(theta_0=np.zeros((n, 1)))
    clf.fit(x_train, y_train)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = clf.predict(x_eval)
    y_pred = y_pred > 0.5
    p = np.where(y_eval == y_pred, 1, 0)
    print('成功率为%d/%d' % (np.sum(p), len(y_eval.flatten())))
    np.savetxt(pred_path, y_pred, fmt='%.1lf')
    plot_decision_boundary(x_train, y_train, clf.theta, save_path=plot_path)
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def g(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        x = np.array(x)
        m, n = x.shape
        y = np.array(y).reshape(m, 1)
        self.theta = np.zeros((n, 1))
        for _ in range(self.max_iter):
            p = self.g(x@self.theta)
            grad = x.T@(p - y)/m
            p = p*(1 - p)
            H = (x.T*p.T)@x/m
            last_theta = self.theta
            self.theta = self.theta - np.linalg.inv(H)@grad*self.step_size
            if np.sum(np.abs(self.theta - last_theta)) < self.eps: break
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        x = np.array(x)
        m, n = x.shape
        y = self.g(x@self.theta)
        return y.flatten()
        # *** END CODE HERE ***
