import numpy as np
from em import MisturaGaussianaDiagonal, NaiveBayesComEM


def test_em_log_verossimilhanca_nao_decresce():
    rng = np.random.default_rng(13)
    X = np.vstack((rng.normal(-2, .6, (150, 2)), rng.normal(2, .8, (150, 2))))
    for k in (2, 3):
        modelo = MisturaGaussianaDiagonal(n_componentes=k, random_state=10).fit(X)
        historico = np.array(modelo.log_verossimilhanca_)
        assert np.all(np.diff(historico) >= -1e-6)
        assert np.isclose(modelo.pesos_.sum(), 1)
        assert np.all(modelo.variancias_ > 0)
        assert np.allclose(modelo.predict_proba(X).sum(axis=1), 1)


def test_nb_com_em_probabilidades_e_classes():
    rng = np.random.default_rng(4)
    X = np.vstack((rng.normal(-2, 1, (100, 3)), rng.normal(2, 1, (100, 3))))
    y = np.array([0] * 100 + [1] * 100)
    for k in (1, 2, 3):
        modelo = NaiveBayesComEM(n_componentes=k).fit(X, y)
        p = modelo.predict_proba(X)
        assert p.shape == (200, 2)
        assert np.allclose(p.sum(axis=1), 1)
        assert np.mean(modelo.predict(X) == y) > .9
