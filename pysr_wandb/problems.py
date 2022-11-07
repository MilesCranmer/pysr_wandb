import numpy as np


def keijzer11(seed):
    """Keijzer-11."""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(-3, 3, size=(20, 2))
    x1 = X[:, 0]
    x2 = X[:, 1]
    y = x1 * x2 + np.sin((x1 - 1) * (x2 - 1))

    return X, y


def keijzer12(seed):
    """Keijzer-12."""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(-3, 3, size=(20, 2))
    x1 = X[:, 0]
    x2 = X[:, 1]
    y = x1**4 - x1**3 + (x2**2 / 2) - x2

    return X, y


def keijzer13(seed):
    """Keijzer-13."""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(-3, 3, size=(20, 2))
    x1 = X[:, 0]
    x2 = X[:, 1]
    y = 6 * np.sin(x1) * np.cos(x2)

    return X, y


def keijzer4(seed):
    """Keijzer-4."""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(0, 10, size=(100, 1))
    x1 = X[:, 0]
    y = (
        x1**3
        * np.exp(-x1)
        * np.cos(x1)
        * np.sin(x1)
        * (np.sin(x1) ** 2 * np.cos(x1) - 1)
    )

    return X, y


def keijzer14(seed):
    """Keijzer-14."""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(-3, 3, size=(20, 2))
    x1 = X[:, 0]
    x2 = X[:, 1]
    y = 8 / (2 + x1**2 + x2**2)

    return X, y


def vlad1(seed):
    """Vladislavleva-1."""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(0.3, 4, size=(100, 2))
    x1 = X[:, 0]
    x2 = X[:, 1]
    y = np.exp(-((x1 - 1) ** 2)) / (1.2 + (x2 - 2.5) ** 2)

    return X, y


def vlad2(seed):
    """Vladislavleva-2."""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(0.05, 10, size=(100, 1))
    x1 = X[:, 0]
    y = (
        np.exp(-x1)
        * x1**3
        * (np.cos(x1) * np.sin(x1))
        * (np.cos(x1) * np.sin(x1) ** 2 - 1)
    )

    return X, y


def vlad3(seed):
    """Vladislavleva-3."""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(0.05, 10, size=(100, 2))
    x1 = X[:, 0]
    x2 = X[:, 1]
    y = (
        np.exp(-x1)
        * x1**3
        * (np.cos(x1) * np.sin(x1))
        * (np.cos(x1) * np.sin(x1) ** 2 - 1)
        * (x2 - 5)
    )

    return X, y


def vlad4(seed):
    """Vladislavleva-4."""
    rstate = np.random.RandomState(seed)
    X = rstate.uniform(0.05, 6.05, size=(1024, 5))
    y = 10 / (5 + np.sum((X - 3) ** 2, axis=1))

    return X, y


def vlad5(seed):
    """Vladislavleva-5."""
    rstate = np.random.RandomState(seed)
    x1 = rstate.uniform(0.05, 2, size=300)
    x2 = rstate.uniform(1, 2, size=300)
    x3 = rstate.uniform(0.05, 2, size=300)
    X = np.vstack((x1, x2, x3)).T
    y = 30 * (x1 - 1) * (x3 - 1) / ((x1 - 10) * x2**2)

    return X, y


PROBLEMS = [
    vlad1,
    vlad2,
    vlad3,
    vlad4,
    vlad5,
    keijzer4,
    keijzer11,
    keijzer12,
    keijzer13,
    keijzer14,
]
