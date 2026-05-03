import numpy as np


class LinearRegression:
    def __init__(
            self,
            *,
            penalty="l2",
            alpha=0.0001,
            max_iter=1000,
            tol=0.001,
            random_state=None,
            eta0=0.01,
            early_stopping=False,
            validation_fraction=0.1,
            n_iter_no_change=5,
            shuffle=True,
            batch_size=32
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_iter_no_change = n_iter_no_change
        self.validation_fraction = validation_fraction
        self.early_stopping = early_stopping
        self.eta0 = eta0
        self.random_state = random_state
        self.tol = tol
        self.max_iter = max_iter
        self.alpha = alpha
        self.penalty = penalty
        self.w_ = None
        self.b_ = None

    def get_penalty_grad(self):
        if self.w_ is None:
            return 0
        if self.penalty == "l1":
            return self.alpha * np.sign(self.w_)
        elif self.penalty == "l2":
            return 2 * self.alpha * self.w_
        else:
            return np.zeros_like(self.w_)

    def fit(self, x, y):
        samples, features = x.shape
        if self.w_ is None:
            self.w_ = np.zeros(features)
        if self.b_ is None:
            self.b_ = 0.

        if self.random_state is not None:
            np.random.seed(self.random_state)

        if self.early_stopping:
            ed_size = int(self.validation_fraction * samples)
            indices = np.random.permutation(samples)
            ed_x = x[indices[:ed_size]]
            ed_y = y[indices[:ed_size]]
            train_x = x[indices[ed_size:]]
            train_y = y[indices[ed_size:]]
            train_samples = len(train_x)
        else:
            ed_x = None
            ed_y = None
            train_x = x
            train_y = y
            train_samples = samples

        loss_h = []
        best_loss = float('inf')
        no_improve_count = 0

        for epoch in range(self.max_iter):
            if self.shuffle:
                indices = np.random.permutation(train_samples)
                train_x = train_x[indices]
                train_y = train_y[indices]

            ep_loss = []

            for start in range(0, train_samples, self.batch_size):
                end = min(train_samples, start + self.batch_size)
                batch_x = train_x[start:end]
                batch_y = train_y[start:end]
                batch_size = len(batch_y)

                y_predict = batch_x @ self.w_ + self.b_
                error = y_predict - batch_y
                batch_loss = np.mean(error**2)
                ep_loss.append(batch_loss)
                grad_w = 2 * (batch_x.T @ error) / batch_size + self.get_penalty_grad()
                grad_b = 2 * np.mean(error)
                self.w_ -= grad_w * self.eta0
                self.b_ -= grad_b * self.eta0

            loss_h.append(np.mean(ep_loss))

            if self.early_stopping:
                ed_predict = self.predict(ed_x)
                ed_loss = np.mean((ed_predict - ed_y)**2)
                if ed_loss < best_loss - self.tol:
                    best_loss = ed_loss
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                if no_improve_count >= self.n_iter_no_change:
                    print(f"Early stopping on {epoch}")
                    break

            elif epoch > 0:
                change = np.abs(loss_h[-1] - loss_h[-2])
                if change < self.tol:
                    print(f"No change stopping on {epoch}")
                    break

        return self

    def predict(self, x):
        if self.w_ is None or self.b_ is None:
            raise AttributeError("Not fitted yet")
        return x @ self.w_ + self.b_

    @property
    def coef_(self):
        return self.w_

    @property
    def intercept_(self):
        return self.b_

    @coef_.setter
    def coef_(self, value):
        self.w_ = value

    @intercept_.setter
    def intercept_(self, value):
        self.b_ = value
