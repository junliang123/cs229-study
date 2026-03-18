import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
 # *** START CODE HERE ***
    # 1. 提前在循环外加载所有数据集（Train, Valid, Test）
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)

    best_tau = None
    min_mse = float('inf')

    # 准备画图：根据 tau 的数量动态创建一个多行2列的子图画布
    n_taus = len(tau_values)
    rows = int(np.ceil(n_taus / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(12, 5 * rows))
    axes = axes.flatten() # 把二维的坐标轴矩阵拉平，方便遍历

    # 2. 遍历所有超参数进行验证
    for i, tau in enumerate(tau_values):
        clf = LocallyWeightedLinearRegression(tau)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_valid)
        
        mse = np.mean((y_pred - y_valid) ** 2)
        print(f"Validation MSE for tau={tau}: {mse}")
        
        # 记录最小的 MSE 和对应的 tau
        if mse < min_mse:
            min_mse = mse
            best_tau = tau

        # 3. 在对应的子图上作图
        axes[i].plot(x_train[:, 1], y_train, 'bx', label='Training Set')
        axes[i].plot(x_valid[:, 1], y_pred, 'ro', label='Validation Predictions')
        axes[i].set_title(f'tau = {tau}, MSE = {mse:.4f}')
        axes[i].legend()

    # 如果 tau 的数量是奇数，隐藏最后一个多余的子图
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # 调整子图间距并保存到指定路径
    plt.tight_layout()
    plt.savefig('output/p05c.png')

    # 4. 使用选出的 best_tau 在测试集上进行最终评估
    print(f"\nBest tau found: {best_tau}")
    clf_best = LocallyWeightedLinearRegression(best_tau)
    clf_best.fit(x_train, y_train)
    y_test_pred = clf_best.predict(x_test)
    test_mse = np.mean((y_test_pred - y_test) ** 2)
    print(f"Test MSE for best tau={best_tau}: {test_mse}")

    # 5. 保存测试集的预测结果到 txt
    np.savetxt(pred_path, y_test_pred)
    # *** END CODE HERE ***
