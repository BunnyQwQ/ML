import numpy as np

from decision_tree import DecisionTreeClassifier, DecisionTreeRegressor


class RandomForestBase:
    def __init__(
            self,
            *,
            n_estimators=100,
            criterion=None,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_impurity_decrease=0.0,
            max_features="sqrt",
            bootstrap=True,
            max_samples=None,
            oob_score=False,
            random_state=None
    ):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.oob_score = oob_score
        self.random_state = random_state
        self.trees = []
        self.oob_score_ = None
        self.n_features_ = None

    def make_tree(self, seed):
        raise NotImplementedError

    def get_oob_score(self, x, y, oob_masks):
        raise NotImplementedError

    def get_n_samples_bootstrap(self, samples):
        if self.max_samples is None:
            return samples
        elif isinstance(self.max_samples, float):
            return max(1, int(self.max_samples * samples))
        else:
            return min(samples, self.max_samples)

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)
        samples, self.n_features_ = x.shape
        self.rng = np.random.RandomState(self.random_state)
        self.trees = []

        n_bootstrap = self.get_n_samples_bootstrap(samples)
        oob_masks = []

        for i in range(self.n_estimators):
            seed = self.rng.randint(0, np.iinfo(np.int32).max)
            tree = self.make_tree(seed)

            if self.bootstrap:
                indices = self.rng.randint(0, samples, size=n_bootstrap)
                mask = np.ones(samples, dtype=bool)
                mask[indices] = False
                oob_masks.append(mask)
                tree.fit(x[indices], y[indices])
            else:
                tree.fit(x, y)

            self.trees.append(tree)

        if self.oob_score and self.bootstrap:
            self.oob_score_ = self.get_oob_score(x, y, oob_masks)

        return self

    @property
    def estimators_(self):
        if not self.trees:
            return []
        return self.trees


class RandomForestClassifier(RandomForestBase):
    def __init__(
            self,
            *,
            n_estimators=100,
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_impurity_decrease=0.0,
            max_features="sqrt",
            bootstrap=True,
            max_samples=None,
            oob_score=False,
            random_state=None
    ):
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            max_features=max_features,
            bootstrap=bootstrap,
            max_samples=max_samples,
            oob_score=oob_score,
            random_state=random_state
        )
        self.classes_ = None
        self.n_classes_ = None

    def make_tree(self, seed):
        return DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            max_features=self.max_features,
            random_state=seed
        )

    def fit(self, x, y):
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        return super().fit(x, y)

    def get_tree_proba(self, tree, x):
        probs = np.zeros((len(x), self.n_classes_))
        columns = np.searchsorted(self.classes_, tree.classes_)
        probs[:, columns] = tree.predict_proba(x)
        return probs

    def predict_proba(self, x):
        if not self.trees:
            raise AttributeError("Not fitted yet")
        x = np.asarray(x, dtype=float)
        probs = np.zeros((len(x), self.n_classes_))
        for tree in self.trees:
            probs += self.get_tree_proba(tree, x)
        return probs / len(self.trees)

    def predict(self, x):
        probs = self.predict_proba(x)
        return self.classes_[np.argmax(probs, axis=1)]

    def get_oob_score(self, x, y, oob_masks):
        probs = np.zeros((len(x), self.n_classes_))
        counts = np.zeros(len(x))
        for tree, mask in zip(self.trees, oob_masks):
            if not np.any(mask):
                continue
            probs[mask] += self.get_tree_proba(tree, x[mask])
            counts[mask] += 1
        used = counts > 0
        predictions = self.classes_[np.argmax(probs[used], axis=1)]
        return np.mean(predictions == y[used])


class RandomForestRegressor(RandomForestBase):
    def __init__(
            self,
            *,
            n_estimators=100,
            criterion="squared_error",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_impurity_decrease=0.0,
            max_features=1.0,
            bootstrap=True,
            max_samples=None,
            oob_score=False,
            random_state=None
    ):
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            max_features=max_features,
            bootstrap=bootstrap,
            max_samples=max_samples,
            oob_score=oob_score,
            random_state=random_state
        )

    def make_tree(self, seed):
        return DecisionTreeRegressor(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            max_features=self.max_features,
            random_state=seed
        )

    def predict(self, x):
        if not self.trees:
            raise AttributeError("Not fitted yet")
        x = np.asarray(x, dtype=float)
        predictions = np.zeros(len(x))
        for tree in self.trees:
            predictions += tree.predict(x)
        return predictions / len(self.trees)

    def get_oob_score(self, x, y, oob_masks):
        predictions = np.zeros(len(x))
        counts = np.zeros(len(x))
        for tree, mask in zip(self.trees, oob_masks):
            if not np.any(mask):
                continue
            predictions[mask] += tree.predict(x[mask])
            counts[mask] += 1
        used = counts > 0
        predictions = predictions[used] / counts[used]
        return 1 - np.sum((y[used] - predictions) ** 2) / np.sum((y[used] - np.mean(y[used])) ** 2)