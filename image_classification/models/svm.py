import numpy as np


class LinearSVM(object):
    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=200, verbose=False):
        """
        Train this linear classifier using stochastic gradient descent.
        """
        num_train, dim = X.shape
        num_classes = (np.max(y) + 1)  # 类别数
        if self.W is None:
            # initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # 随机梯度下降优化 W
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            indices = np.random.choice(num_train, batch_size, replace=True)

            # 训练集 mini-batch
            X_batch, y_batch = X[indices], y[indices]

            # evaluate loss and gradient
            loss, grad = self.loss_vectorized(self.W, X_batch, y_batch, reg)
            loss_history.append(loss)

            # 更新权重 W
            self.W -= learning_rate * grad

            if verbose and it % 100 == 0:
                print("iteration %d / %d: loss %f" % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for data points.
        """
        y_pred = np.zeros(X.shape[0])
        y_pred = X.dot(self.W).argmax(axis=1)
        return y_pred

    def loss_vectorized(self, W, X, y, reg):
        """
        Structured SVM loss function, vectorized implementation.
        """
        loss = 0.0
        dW = np.zeros(W.shape)  # initialize the gradient as zero

        # 计算 loss
        num_classes = W.shape[1]
        num_train = X.shape[0]
        loss = 0.0
        scores = X.dot(W)  # dims: N x C

        correct_class_scores = scores[np.arange(num_train), y].reshape(-1, 1)

        margin = np.maximum(0, scores - correct_class_scores + 1)  # delta = 1
        margin = np.where(margin == 1.0, 0, margin)

        loss = np.sum(margin)
        average_loss = loss / num_train
        # 正则化损失
        regularized_loss = average_loss + reg * np.sum(W * W)

        valid_margin_mask = np.zeros(margin.shape)  # margin.shape: N x C
        valid_margin_mask[margin > 0] = 1  # if margin is positive, set a positive mask

        valid_margin_mask[np.arange(num_train), y] = -np.sum(valid_margin_mask, axis=1)

        # X.T x valid_margin_mask = (D x N) x (N x C) = D x C
        dW = X.T.dot(valid_margin_mask)

        # 平均梯度
        average_dW = dW / num_train

        # 使用权重正则化梯度
        regularized_dW = average_dW + (2 * reg * W)

        return regularized_loss, regularized_dW
