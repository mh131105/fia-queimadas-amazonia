import unittest
import numpy as np
from aprendizado_probabilistico import mle_bernoulli, map_bernoulli, BernoulliNaiveBayes

class TestProbabilidade(unittest.TestCase):
    def test_estimadores(self):
        y = [1, 1, 1, 0]
        self.assertAlmostEqual(mle_bernoulli(y), 0.75)
        self.assertAlmostEqual(map_bernoulli(y, 2, 2), 4/6)

    def test_extremos_map(self):
        self.assertAlmostEqual(map_bernoulli([0,0,0], 2,2), 1/5)
        self.assertAlmostEqual(map_bernoulli([1,1,1], 2,2), 4/5)

    def test_nb(self):
        X = np.array([[0,0],[0,1],[1,0],[1,1],[1,1],[0,0]])
        y = np.array([0,0,1,1,1,0])
        model = BernoulliNaiveBayes().fit(X,y)
        probs = model.predict_proba(X)
        self.assertTrue(np.allclose(probs.sum(axis=1), 1))
        self.assertEqual(model.predict(np.array([[0,0],[1,1]])).tolist(), [0,1])

if __name__ == '__main__':
    unittest.main()
