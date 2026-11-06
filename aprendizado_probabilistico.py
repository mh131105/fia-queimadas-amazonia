"""Etapa 4: estimadores Bernoulli e Naive Bayes, sem estimadores prontos."""
import numpy as np


def _vetor_binario(y):
    y = np.asarray(y)
    if y.ndim != 1 or y.size == 0 or not np.isin(y, [0, 1]).all():
        raise ValueError('y deve ser vetor não vazio de zeros e uns.')
    return y.astype(np.float64)


def mle_bernoulli(y):
    """Máxima verossimilhança: proporção de sucessos."""
    return float(_vetor_binario(y).mean())


def map_bernoulli(y, alpha=2.0, beta=2.0):
    """Moda posterior com priori Beta(alpha, beta)."""
    y = _vetor_binario(y)
    if alpha <= 0 or beta <= 0:
        raise ValueError('alpha e beta devem ser positivos.')
    a_post, b_post = alpha + y.sum(), beta + y.size - y.sum()
    if a_post > 1 and b_post > 1:
        return float((a_post - 1) / (a_post + b_post - 2))
    if a_post <= 1 < b_post:
        return 0.0
    if b_post <= 1 < a_post:
        return 1.0
    raise ValueError('A posteriori não tem moda única (Beta com ambos parâmetros <= 1).')


class BernoulliNaiveBayes:
    """Bernoulli NB do zero, com suavização aditiva e cálculo em log-space."""
    def __init__(self, alpha=1.0):
        if alpha <= 0:
            raise ValueError('alpha deve ser positivo.')
        self.alpha = float(alpha)

    def fit(self, X, y):
        X = np.asarray(X)
        y = _vetor_binario(y).astype(int)
        if X.ndim != 2 or X.shape[0] != len(y) or X.shape[1] == 0:
            raise ValueError('X deve ser matriz 2D compatível com y.')
        if not np.isin(X, [0, 1]).all():
            raise ValueError('X deve conter apenas atributos binários.')
        X = X.astype(float)
        counts = np.bincount(y, minlength=2)
        self.log_prior_ = np.log((counts + self.alpha) / (len(y) + 2 * self.alpha))
        self.p_ = np.vstack([
            (X[y == c].sum(axis=0) + self.alpha) / (counts[c] + 2 * self.alpha)
            for c in (0, 1)
        ])
        return self

    def predict_log_proba(self, X):
        if not hasattr(self, 'p_'):
            raise ValueError('Ajuste o modelo antes de prever.')
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.p_.shape[1] or not np.isin(X, [0, 1]).all():
            raise ValueError('X incompatível com o treinamento.')
        X = X.astype(float)
        log_joint = (X @ np.log(self.p_).T + (1 - X) @ np.log1p(-self.p_).T
                     + self.log_prior_)
        m = log_joint.max(axis=1, keepdims=True)
        return log_joint - (m + np.log(np.exp(log_joint - m).sum(axis=1, keepdims=True)))

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
