import numpy as np


class SoftmaxRegression:
    def __init__(
            self,
            *,
            penalty="l2",
            alpha=0.0001,
            max_iter=100,
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
        self.n_features_ = None
        self.n_classes_ = None

    def get_penalty_grad(self):
        if self.w_ is None:
            return 0
        if self.penalty == "l1":
            return self.alpha * np.sign(self.w_)
        elif self.penalty == "l2":
            return self.alpha * self.w_ * 2
        else:
            return np.zeros_like(self.w_)

    def fit(self, x, y):
        self.n_classes_ = len(np.unique(y))
        samples, self.n_features_ = x.shape
        self.rng = np.random.RandomState(self.random_state)
        if self.w_ is None:
            self.w_ = self.rng.randn(self.n_classes_, self.n_features_) * 0.01
        if self.b_ is None:
            self.b_ = np.zeros(self.n_classes_)

        if self.early_stopping:
            ed_size = int(self.validation_fraction * samples)
            indices = self.rng.permutation(samples)
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
        eps = 1e-15
        learning_rate = self.eta0 * 3
        for epoch in range(self.max_iter):
            if self.shuffle:
                indices = self.rng.permutation(train_samples)
                shuffled_x = train_x[indices]
                shuffled_y = train_y[indices]
            else:
                shuffled_x = train_x
                shuffled_y = train_y

            ep_loss = []

            for start in range(0, train_samples, self.batch_size):
                end = min(train_samples, start + self.batch_size)
                batch_x = shuffled_x[start:end]
                batch_y = shuffled_y[start:end]
                batch_size = len(batch_y)
                probs = self.predict_proba(batch_x)
                y_onehot = np.zeros((batch_size, self.n_classes_))
                y_onehot[np.arange(batch_size), batch_y] = 1
                error = probs - y_onehot
                batch_loss = -np.mean(np.sum(y_onehot * np.log(probs + eps), axis=1))
                ep_loss.append(batch_loss)
                grad_w = (batch_x.T @ error).T / batch_size + self.get_penalty_grad()
                grad_b = np.mean(error, axis=0)
                self.w_ -= grad_w * learning_rate
                self.b_ -= grad_b * learning_rate

            loss_h.append(np.mean(ep_loss))

            if self.early_stopping:
                ed_probs = self.predict_proba(ed_x)
                ed_y_onehot = np.zeros((ed_size, self.n_classes_))
                ed_y_onehot[np.arange(ed_size), ed_y] = 1
                ed_loss = -np.mean(np.sum(ed_y_onehot * np.log(ed_probs + eps), axis=1))
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
                if change < self.tol * 0.1:
                    print(f"No change stopping on {epoch}")
                    break

        return self

    def predict_proba(self, x):
        z = x @ self.w_.T + self.b_
        return self.softmax(z)

    def predict(self, x):
        proba = self.predict_proba(x)
        if proba.ndim == 1:
            return np.argmax(proba)
        else:
            return np.argmax(proba, axis=1)

    @staticmethod
    def softmax(z) -> np.ndarray:
        if z.ndim == 1:
            zwithoutmax = z - np.max(z)
            exp_z = np.exp(zwithoutmax)
            return exp_z / np.sum(exp_z)
        else:
            zwithoutmax = z - np.max(z, axis=1, keepdims=True)
            exp_z = np.exp(zwithoutmax)
            return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    @property
    def coef_(self):
        return self.w_

    @property
    def intercept_(self):
        return self.b_

    @coef_.setter
    def coef_(self, value):
        self.w_ = value
        if value is not None:
            self.n_classes_, self.n_features_ = value.shape

    @intercept_.setter
    def intercept_(self, value):
        self.b_ = value
