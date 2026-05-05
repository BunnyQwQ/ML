import numpy as np
from sklearn.tree import DecisionTreeRegressor


class GBCustomRegressor:
    def __init__(
            self,
            *,
            learning_rate=0.1,
            n_estimators=100,
            criterion="friedman_mse",
            min_samples_split=2,
            min_samples_leaf=1,
            max_depth=3,
            random_state=None
    ):
        self.trees = []
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.random_state = random_state
        self.start_median = None

    def fit(self, x, y):
        self.trees = []
        self.start_median = np.mean(y)
        cur_pred = np.ones_like(y) * self.start_median
        for i in range(self.n_estimators):
            tree = DecisionTreeRegressor(
                criterion=self.criterion,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
            diff = y - cur_pred
            tree.fit(x, diff)
            self.trees.append(tree)
            cur_pred += self.learning_rate * tree.predict(x)

    def predict(self, x):
        if not self.trees:
            return np.array([])
        predictions = np.full(x.shape[0], self.start_median)
        for tree in self.trees:
            predictions += tree.predict(x) * self.learning_rate
        return predictions

    @property
    def estimators_(self):
        if not hasattr(self, 'trees') or len(self.trees) == 0:
            return []
        return self.trees


class GBCustomClassifier:
    def __init__(
            self,
            *,
            learning_rate=0.1,
            n_estimators=100,
            criterion="friedman_mse",
            min_samples_split=2,
            min_samples_leaf=1,
            max_depth=3,
            random_state=None
    ):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []
        self.start_constant = None

    def fit(self, x, y):
        self.trees = []
        p = np.mean(y)
        self.start_constant = np.log(p / (1 - p))
        cur_logits = np.ones_like(y) * self.start_constant
        for i in range(self.n_estimators):
            prob = 1 / (1 + np.exp(-cur_logits))
            diff = y - prob
            tree = DecisionTreeRegressor(
                criterion=self.criterion,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
            tree.fit(x, diff)
            self.trees.append(tree)
            cur_logits += self.learning_rate * tree.predict(x)

    def predict_proba(self, x):
        logits = np.full(x.shape[0], self.start_constant)
        for tree in self.trees:
            logits += self.learning_rate * tree.predict(x)
        prob = 1 / (1 + np.exp(-logits))
        return np.column_stack((1 - prob, prob))

    def predict(self, x):
        probs = self.predict_proba(x)[:, 1]
        if probs.size == 0:
            return np.array([])

        return (probs >= 0.5).astype(int)

    @property
    def estimators_(self):
        if not hasattr(self, 'trees') or len(self.trees) == 0:
            return []
        return self.trees
