import numpy as np


class SVM:
    def __init__(
            self,
            *,
            penalty="l2",
            alpha=0.0001,
            C=1.0,
            loss="hinge",
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
        self.loss = loss
        self.C = C
        self.alpha = alpha
        self.penalty = penalty
        self.w_ = None
        self.b_ = None
        self.classes_ = None

    def get_penalty_grad(self):
        if self.w_ is None:
            return 0
        if self.penalty == "l1":
            return self.alpha * np.sign(self.w_)
        elif self.penalty == "l2":
            return 2 * self.alpha * self.w_
        else:
            return np.zeros_like(self.w_)

    def get_penalty_loss(self):
        if self.w_ is None:
            return 0.
        if self.penalty == "l1":
            return self.alpha * np.sum(np.abs(self.w_))
        elif self.penalty == "l2":
            return self.alpha * np.sum(self.w_ ** 2)
        else:
            return 0.

    def get_loss(self, x, y):
        margin = y * (x @ self.w_ + self.b_)
        slack = np.maximum(0, 1 - margin)
        if self.loss == "squared_hinge":
            slack = slack ** 2
        return self.C * np.mean(slack) + self.get_penalty_loss()

    def fit(self, x, y):
        samples, features = x.shape
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("Only binary classification is supported")
        y_signed = np.where(y == self.classes_[0], -1., 1.)

        self.rng = np.random.RandomState(self.random_state)
        if self.w_ is None:
            self.w_ = np.zeros(features)
        if self.b_ is None:
            self.b_ = 0.

        if self.early_stopping:
            ed_size = int(self.validation_fraction * samples)
            indices = self.rng.permutation(samples)
            ed_x = x[indices[:ed_size]]
            ed_y = y_signed[indices[:ed_size]]
            train_x = x[indices[ed_size:]]
            train_y = y_signed[indices[ed_size:]]
            train_samples = len(train_x)
        else:
            ed_x = None
            ed_y = None
            train_x = x
            train_y = y_signed
            train_samples = samples

        loss_h = []
        best_loss = float('inf')
        no_improve_count = 0

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

                margin = batch_y * (batch_x @ self.w_ + self.b_)
                slack = np.maximum(0, 1 - margin)

                if self.loss == "squared_hinge":
                    batch_loss = self.C * np.mean(slack ** 2) + self.get_penalty_loss()
                    coefs = -2 * self.C * slack * batch_y
                else:
                    batch_loss = self.C * np.mean(slack) + self.get_penalty_loss()
                    coefs = -self.C * (slack > 0) * batch_y

                ep_loss.append(batch_loss)
                grad_w = batch_x.T @ coefs / batch_size + self.get_penalty_grad()
                grad_b = np.mean(coefs)
                self.w_ -= grad_w * self.eta0
                self.b_ -= grad_b * self.eta0

            loss_h.append(np.mean(ep_loss))

            if self.early_stopping:
                ed_loss = self.get_loss(ed_x, ed_y)
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

    def decision_function(self, x):
        if self.w_ is None or self.b_ is None:
            raise AttributeError("Not fitted yet")
        return x @ self.w_ + self.b_

    def predict(self, x):
        scores = self.decision_function(x)
        return np.where(scores >= 0, self.classes_[1], self.classes_[0])

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