import numpy as np
import util
import matplotlib.pyplot as plt
from scipy import stats

from linear_model import LinearModel

def plot_gda_decision_boundary(x, y, theta, theta0, save_path="gda_output.png"):
    """
    绘制数据散点图和 GDA 的决策边界
    注意：这里的 x 没有截距项 (x[:, 0] 是 x1, x[:, 1] 是 x2)
    """
    plt.figure(figsize=(8, 6))
    
    # 将 y 展平以匹配索引
    y = y.flatten()
    
    # 1. 绘制散点图
    plt.plot(x[y == 0, 0], x[y == 0, 1], 'bx', label='y=0')
    plt.plot(x[y == 1, 0], x[y == 1, 1], 'go', label='y=1')
    
    # 2. 准备横坐标 x1 的范围
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
    x1_vals = np.array([x1_min - 0.5, x1_max + 0.5])
    
    # 3. 计算对应的纵坐标 x2
    theta = theta.flatten()
    theta0 = float(theta0) # 确保 theta0 是标量
    
    x2_vals = -(theta[0] * x1_vals + theta0) / theta[1]
    
    # 4. 绘制决策边界
    plt.plot(x1_vals, x2_vals, 'k-', linewidth=2, label='GDA')
    
    # 设置图表属性
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Dataset: Gaussian Discriminant Analysis')
    plt.legend()
    
    # 保存并关闭图片，防止程序阻塞和画图重叠
    plt.savefig(save_path)
    plt.close()

def main(train_path, eval_path, pred_path, plot_path=None):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
        plot_path: Path to save plot.
    """
    # Load dataset
    # *** START CODE HERE ***
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)
    
    # 附加题：对 Dataset 2 的特征 2 (x2) 应用平移 + Box-Cox 变换
    if 'ds1' in train_path:
        x_train_transformed = np.copy(x_train)
        x_eval_transformed = np.copy(x_eval)
        
        # 寻找偏移量，确保所有数据严格大于 0
        min_val = np.min(x_train[:, 1])
        offset = 0 if min_val > 0 else (abs(min_val) + 1e-4)
        
        # 对训练集进行平移和变换
        x_train_shifted = x_train[:, 1] + offset
        x_train_transformed[:, 1], lmbda = stats.boxcox(x_train_shifted)
        
        # 必须对评估集使用【相同的偏移量】和【相同的 lambda】
        x_eval_shifted = x_eval[:, 1] + offset
        x_eval_transformed[:, 1] = stats.boxcox(x_eval_shifted, lmbda=lmbda)
        
        x_train = x_train_transformed
        x_eval = x_eval_transformed
        print(f"检测到 Dataset 2，应用 Box-Cox 变换 (lambda: {lmbda:.4f}, offset: {offset:.4f})")

    clf = GDA()
    clf.fit(x_train, y_train)
    
    y_pred = clf.predict(x_eval)
    
    # 准确率计算
    N = np.sum(y_eval.flatten() == y_pred)
    M = len(y_eval.flatten())
    print('成功率为%d/%d (%.2f%%)' % (N, M, 100 * N / M))
    
    np.savetxt(pred_path, y_pred, fmt='%d') # 0或1，保存为整数更干净
    
    if plot_path is not None:
        plot_gda_decision_boundary(x_train, y_train, clf.theta, clf.theta0, save_path=plot_path)
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        x = np.array(x)
        m, n = x.shape
        y = np.array(y).reshape(m, 1)
        
        phi = np.sum(y == 1) / m
        u0 = x.T @ (1 - y) / np.sum(y == 0)
        u1 = x.T @ y / np.sum(y == 1)
        
        # 向量化计算协方差矩阵
        mu_y = np.where(y == 0, u0.T, u1.T) 
        diff = x - mu_y 
        sigma = (diff.T @ diff) / m
        
        # 计算 theta 和 theta0 (修正：u1 - u0)
        sigma_inv = np.linalg.inv(sigma)
        self.theta = sigma_inv @ (u1 - u0)
        self.theta0 = (u0 + u1).T @ sigma_inv @ (u0 - u1) / 2.0
        self.theta0 = self.theta0 - np.log((1 - phi) / phi)
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
        
        # 计算线性边界: z = x*theta + theta0
        z = x @ self.theta + self.theta0
        
        # Sigmoid 函数求概率
        h_x = 1 / (1 + np.exp(-z))
        
        # 转换为 0 或 1，并展平为一维数组
        y_pred = (h_x > 0.5).astype(int).flatten()
        return y_pred
        # *** END CODE HERE ***