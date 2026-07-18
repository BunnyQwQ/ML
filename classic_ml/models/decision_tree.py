import numpy as np


class Node:
    __slots__ = ("feature", "threshold", "left", "right", "value", "n_samples")

    def __init__(self, *, feature=None, threshold=None, left=None, right=None, value=None, n_samples=0):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.n_samples = n_samples

    @property
    def is_leaf(self):
        return self.value is not None


class DecisionTreeBase:
    def __init__(
            self,
            *,
            criterion="squared_error",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_impurity_decrease=0.0,
            max_features=None,
            random_state=None
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.random_state = random_state
        self.root_ = None
        self.n_features_ = None

    def get_n_features_split(self):
        if self.max_features is None:
            return self.n_features_
        elif self.max_features == "sqrt":
            return max(1, int(np.sqrt(self.n_features_)))
        elif self.max_features == "log2":
            return max(1, int(np.log2(self.n_features_)))
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * self.n_features_))
        else:
            return min(self.n_features_, self.max_features)

    def get_split_costs(self, sorted_y):
        raise NotImplementedError

    def get_node_cost(self, y):
        raise NotImplementedError

    def get_leaf_value(self, y):
        raise NotImplementedError

    def get_split(self, x, y):
        samples = len(y)
        n_split = self.get_n_features_split()
        features = self.rng.choice(self.n_features_, size=n_split, replace=False)

        best_feature = None
        best_threshold = None
        best_cost = np.inf

        for feature in features:
            order = np.argsort(x[:, feature], kind="mergesort")
            xs = x[order, feature]
            ys = y[order]

            costs = self.get_split_costs(ys)
            valid = xs[:-1] < xs[1:]
            if self.min_samples_leaf > 1:
                left_sizes = np.arange(1, samples)
                valid &= (left_sizes >= self.min_samples_leaf)
                valid &= (samples - left_sizes >= self.min_samples_leaf)
            if not np.any(valid):
                continue

            costs = np.where(valid, costs, np.inf)
            position = np.argmin(costs)
            if costs[position] < best_cost:
                best_cost = costs[position]
                best_feature = feature
                best_threshold = (xs[position] + xs[position + 1]) / 2

        return best_feature, best_threshold, best_cost

    def build_tree(self, x, y, depth):
        samples = len(y)
        if (
                samples < self.min_samples_split
                or samples < 2 * self.min_samples_leaf
                or (self.max_depth is not None and depth >= self.max_depth)
                or self.get_node_cost(y) <= 0
        ):
            return Node(value=self.get_leaf_value(y), n_samples=samples)

        feature, threshold, cost = self.get_split(x, y)
        if feature is None:
            return Node(value=self.get_leaf_value(y), n_samples=samples)

        decrease = (self.get_node_cost(y) - cost) / self.n_samples_total_
        if decrease < self.min_impurity_decrease:
            return Node(value=self.get_leaf_value(y), n_samples=samples)

        mask = x[:, feature] <= threshold
        left = self.build_tree(x[mask], y[mask], depth + 1)
        right = self.build_tree(x[~mask], y[~mask], depth + 1)
        return Node(
            feature=feature,
            threshold=threshold,
            left=left,
            right=right,
            n_samples=samples
        )

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = self.prepare_y(y)
        _, self.n_features_ = x.shape
        self.n_samples_total_ = len(y)
        self.rng = np.random.RandomState(self.random_state)
        self.root_ = self.build_tree(x, y, 0)
        return self

    def prepare_y(self, y):
        return np.asarray(y, dtype=float)

    def apply_row(self, row):
        node = self.root_
        while not node.is_leaf:
            if row[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def get_leaf_values(self, x):
        if self.root_ is None:
            raise AttributeError("Not fitted yet")
        x = np.asarray(x, dtype=float)
        return np.array([self.apply_row(row) for row in x])

    def get_depth(self):
        def depth_of(node):
            if node.is_leaf:
                return 0
            return 1 + max(depth_of(node.left), depth_of(node.right))
        return depth_of(self.root_)


class DecisionTreeRegressor(DecisionTreeBase):
    def get_node_cost(self, y):
        return np.sum((y - np.mean(y)) ** 2)

    def get_split_costs(self, sorted_y):
        samples = len(sorted_y)
        cum_y = np.cumsum(sorted_y)[:-1]
        cum_y2 = np.cumsum(sorted_y ** 2)[:-1]
        total_y = cum_y[-1] + sorted_y[-1]
        total_y2 = cum_y2[-1] + sorted_y[-1] ** 2

        n_left = np.arange(1, samples)
        n_right = samples - n_left
        sse_left = cum_y2 - cum_y ** 2 / n_left
        sse_right = (total_y2 - cum_y2) - (total_y - cum_y) ** 2 / n_right
        return sse_left + sse_right

    def get_leaf_value(self, y):
        return np.mean(y)

    def predict(self, x):
        return self.get_leaf_values(x)


class DecisionTreeClassifier(DecisionTreeBase):
    def __init__(
            self,
            *,
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_impurity_decrease=0.0,
            max_features=None,
            random_state=None
    ):
        super().__init__(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            max_features=max_features,
            random_state=random_state
        )
        self.classes_ = None
        self.n_classes_ = None

    def prepare_y(self, y):
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        return encoded

    def get_impurity(self, counts, sizes):
        if self.criterion == "entropy":
            eps = 1e-15
            probs = counts / sizes
            return -np.sum(counts * np.log2(probs + eps), axis=1)
        return sizes[:, 0] - np.sum(counts ** 2, axis=1) / sizes[:, 0]

    def get_node_cost(self, y):
        counts = np.bincount(y, minlength=self.n_classes_).reshape(1, -1).astype(float)
        sizes = np.array([[len(y)]], dtype=float)
        return self.get_impurity(counts, sizes)[0]

    def get_split_costs(self, sorted_y):
        samples = len(sorted_y)
        onehot = np.zeros((samples, self.n_classes_))
        onehot[np.arange(samples), sorted_y] = 1

        cum_counts = np.cumsum(onehot, axis=0)[:-1]
        total_counts = cum_counts[-1] + onehot[-1]
        counts_left = cum_counts
        counts_right = total_counts - cum_counts

        n_left = np.arange(1, samples).reshape(-1, 1).astype(float)
        n_right = samples - n_left
        return self.get_impurity(counts_left, n_left) + self.get_impurity(counts_right, n_right)

    def get_leaf_value(self, y):
        counts = np.bincount(y, minlength=self.n_classes_).astype(float)
        return counts / np.sum(counts)

    def predict_proba(self, x):
        return self.get_leaf_values(x)

    def predict(self, x):
        probs = self.predict_proba(x)
        return self.classes_[np.argmax(probs, axis=1)]