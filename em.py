"""EM do zero para misturas gaussianas com covariância diagonal.

Inclui um classificador supervisionado de misturas por classe: K=1 é o
Gaussian Naive Bayes, K=2/3 é Naive Bayes com perfis latentes via EM.
"""

import numpy as np


def _as_matrix(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not X.shape[0] or not X.shape[1]:
        raise ValueError("X deve ser uma matriz não vazia (n_amostras, n_atributos).")
    if not np.all(np.isfinite(X)):
        raise ValueError("X deve conter somente valores finitos.")
    return X


def _logsumexp(A, axis=1):
    m = np.max(A, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(A - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


class MisturaGaussianaDiagonal:
    def __init__(self, n_componentes=2, max_iter=100, tol=1e-5,
                 reg_covar=1e-4, random_state=42):
        if n_componentes < 1 or max_iter < 1 or reg_covar <= 0 or tol < 0:
            raise ValueError("Parâmetros inválidos.")
        self.n_componentes = int(n_componentes)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.reg_covar = float(reg_covar)
        self.random_state = random_state

    def _log_prob_componentes(self, X):
        # Log-Normal multivariada com matriz de covariância diagonal.
        d = X.shape[1]
        diff = X[:, None, :] - self.medias_[None, :, :]
        term = np.sum(np.log(self.variancias_), axis=1)
        quad = np.sum(diff * diff / self.variancias_[None, :, :], axis=2)
        return -0.5 * (d * np.log(2.0 * np.pi) + term[None, :] + quad)

    def _log_joint(self, X):
        return self._log_prob_componentes(X) + np.log(self.pesos_)[None, :]

    def fit(self, X):
        X = _as_matrix(X)
        n, d = X.shape
        K = self.n_componentes
        if n < K:
            raise ValueError("Número de amostras inferior ao de componentes.")
        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(n, size=K, replace=False)
        self.medias_ = X[indices].copy()
        variancia_global = np.maximum(np.var(X, axis=0), self.reg_covar)
        self.variancias_ = np.broadcast_to(variancia_global, (K, d)).copy()
        self.pesos_ = np.full(K, 1.0 / K)
        self.log_verossimilhanca_ = []
        self.convergiu_ = False

        # A primeira entrada reflete os parâmetros iniciais (iteração zero).
        log_anterior = float(np.sum(_logsumexp(self._log_joint(X))))
        self.log_verossimilhanca_.append(log_anterior)

        for _ in range(self.max_iter):
            joint = self._log_joint(X)
            log_norm = _logsumexp(joint)
            resp = np.exp(joint - log_norm[:, None])

            Nk = resp.sum(axis=0)
            # Com dados finitos, resp é finita, mas algum componente pode colapsar.
            Nk_seguro = np.maximum(Nk, np.finfo(float).tiny)
            self.pesos_ = Nk_seguro / Nk_seguro.sum()
            self.medias_ = (resp.T @ X) / Nk_seguro[:, None]
            diff = X[:, None, :] - self.medias_[None, :, :]
            self.variancias_ = np.einsum('nk,nkd->kd', resp, diff * diff) / Nk_seguro[:, None]
            self.variancias_ = np.maximum(self.variancias_, self.reg_covar)

            log_atual = float(np.sum(_logsumexp(self._log_joint(X))))
            self.log_verossimilhanca_.append(log_atual)
            ganho = log_atual - log_anterior
            if abs(ganho) <= self.tol * (1.0 + abs(log_anterior)):
                self.convergiu_ = True
                break
            log_anterior = log_atual

        self.n_iter_ = len(self.log_verossimilhanca_) - 1
        return self

    def score_samples(self, X):
        X = _as_matrix(X)
        if X.shape[1] != self.medias_.shape[1]:
            raise ValueError("Dimensão de atributos incompatível.")
        return _logsumexp(self._log_joint(X))

    def predict_proba(self, X):
        X = _as_matrix(X)
        joint = self._log_joint(X)
        return np.exp(joint - _logsumexp(joint)[:, None])

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class NaiveBayesComEM:
    """Mistura gaussiana diagonal condicionada à classe.

    Para n_componentes=1, reproduz NB gaussiano (com piso de variância).
    Para n_componentes>1, utiliza EM em X[y == classe] e marginaliza Z.
    """
    def __init__(self, n_componentes=1, max_iter=100, tol=1e-5,
                 reg_covar=1e-4, random_state=42):
        self.n_componentes = n_componentes
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state

    def fit(self, X, y):
        X = _as_matrix(X)
        y = np.asarray(y).reshape(-1)
        if len(y) != len(X):
            raise ValueError("X e y devem ter o mesmo número de amostras.")
        self.classes_, contagens = np.unique(y, return_counts=True)
        if len(self.classes_) != 2:
            raise ValueError("Este experimento espera exatamente duas classes.")
        self.prioris_ = contagens / len(y)
        self.modelos_ = []
        for i, classe in enumerate(self.classes_):
            modelo = MisturaGaussianaDiagonal(
                n_componentes=self.n_componentes,
                max_iter=self.max_iter,
                tol=self.tol,
                reg_covar=self.reg_covar,
                random_state=self.random_state + i,
            ).fit(X[y == classe])
            self.modelos_.append(modelo)
        return self

    def predict_proba(self, X):
        X = _as_matrix(X)
        scores = np.column_stack([
            np.log(priori) + modelo.score_samples(X)
            for priori, modelo in zip(self.prioris_, self.modelos_)
        ])
        return np.exp(scores - _logsumexp(scores)[:, None])

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
