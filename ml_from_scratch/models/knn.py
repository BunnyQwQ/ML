import numpy as np


class KNNBase:
    def __init__(
            self,
            *,
            n_neighbors=5,
            weights="uniform",
            p=2,
            batch_size=512
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        self.batch_size = batch_size
        self.x_ = None
        self.y_ = None
        self.n_features_ = None

    def fit(self, x, y):
        self.x_ = np.asarray(x, dtype=float)
        self.y_ = np.asarray(y)
        _, self.n_features_ = self.x_.shape
        return self

    def get_distances(self, batch_x):
        if self.p == 2:
            sq = (
                np.sum(batch_x ** 2, axis=1, keepdims=True)
                - 2 * batch_x @ self.x_.T
                + np.sum(self.x_ ** 2, axis=1)
            )
            return np.sqrt(np.maximum(sq, 0))
        diff = np.abs(batch_x[:, None, :] - self.x_[None, :, :])
        if self.p == 1:
            return np.sum(diff, axis=2)
        elif self.p == np.inf:
            return np.max(diff, axis=2)
        else:
            return np.sum(diff ** self.p, axis=2) ** (1 / self.p)

    def get_neighbors(self, batch_x):
        distances = self.get_distances(batch_x)
        k = min(self.n_neighbors, self.x_.shape[0])
        indices = np.argpartition(distances, k - 1, axis=1)[:, :k]
        neighbor_dist = np.take_along_axis(distances, indices, axis=1)
        order = np.argsort(neighbor_dist, axis=1)
        indices = np.take_along_axis(indices, order, axis=1)
        neighbor_dist = np.take_along_axis(neighbor_dist, order, axis=1)
        return neighbor_dist, indices

    def get_weights(self, neighbor_dist):
        if self.weights == "distance":
            eps = 1e-15
            w = 1 / (neighbor_dist + eps)
            exact = neighbor_dist < eps
            rows = np.any(exact, axis=1)
            w[rows] = exact[rows].astype(float)
            return w / np.sum(w, axis=1, keepdims=True)
        return np.full(neighbor_dist.shape, 1 / neighbor_dist.shape[1])

    def kneighbors(self, x):
        if self.x_ is None:
            raise AttributeError("Not fitted yet")
        x = np.asarray(x, dtype=float)
        all_dist = []
        all_idx = []
        for start in range(0, len(x), self.batch_size):
            end = min(len(x), start + self.batch_size)
            neighbor_dist, indices = self.get_neighbors(x[start:end])
            all_dist.append(neighbor_dist)
            all_idx.append(indices)
        return np.vstack(all_dist), np.vstack(all_idx)


class KNNClassifier(KNNBase):
    def __init__(
            self,
            *,
            n_neighbors=5,
            weights="uniform",
            p=2,
            batch_size=512
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            weights=weights,
            p=p,
            batch_size=batch_size
        )
        self.classes_ = None

    def fit(self, x, y):
        super().fit(x, y)
        self.classes_, self.y_ = np.unique(self.y_, return_inverse=True)
        return self

    def predict_proba(self, x):
        if self.x_ is None:
            raise AttributeError("Not fitted yet")
        neighbor_dist, indices = self.kneighbors(x)
        w = self.get_weights(neighbor_dist)
        labels = self.y_[indices]
        probs = np.zeros((len(labels), len(self.classes_)))
        for cls in range(len(self.classes_)):
            probs[:, cls] = np.sum(w * (labels == cls), axis=1)
        return probs

    def predict(self, x):
        probs = self.predict_proba(x)
        return self.classes_[np.argmax(probs, axis=1)]


class KNNRegressor(KNNBase):
    def predict(self, x):
        if self.x_ is None:
            raise AttributeError("Not fitted yet")
        neighbor_dist, indices = self.kneighbors(x)
        w = self.get_weights(neighbor_dist)
        targets = self.y_[indices]
        if targets.ndim == 3:
            return np.sum(w[:, :, None] * targets, axis=1)
        return np.sum(w * targets, axis=1)