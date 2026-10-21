import unittest
from app import *
class DomainTests(unittest.TestCase):
    def test_piecewise_signal(self):
        x=np.arange(100).reshape(-1,1);y=(x[:,0]>49).astype(float);self.assertLess(rmse(y,predict(fit(x,y,3,5),x)),.2)
    def test_constant_leaf(self):
        tree=fit(np.ones((30,2)),np.ones(30)*7);np.testing.assert_allclose(predict(tree,np.zeros((3,2))),7)
    def test_depth_zero(self):self.assertNotIn('feature',fit(np.arange(50).reshape(-1,1),np.arange(50),0))
