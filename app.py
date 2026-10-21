"""Train a depth-limited regression tree on nonlinear synthetic demand and inspect the learned rules."""
from kit import *
TITLE='Interpretable Nonlinear Regression'
LIVE=False

def fit(x,y,depth=4,min_leaf=12):
    node={'value':float(y.mean())}
    if depth==0 or len(y)<2*min_leaf:return node
    best=None;loss=float(np.sum((y-y.mean())**2))
    for j in range(x.shape[1]):
        for t in np.unique(np.quantile(x[:,j],np.linspace(.1,.9,9))):
            mask=x[:,j]<=t
            if min(mask.sum(),(~mask).sum())<min_leaf:continue
            value=float(np.sum((y[mask]-y[mask].mean())**2)+np.sum((y[~mask]-y[~mask].mean())**2))
            if value<loss-1e-9:loss=value;best=j,float(t),mask
    if best:
        j,t,mask=best;node.update(feature=j,threshold=t,left=fit(x[mask],y[mask],depth-1,min_leaf),right=fit(x[~mask],y[~mask],depth-1,min_leaf))
    return node
def predict(tree,x):
    def one(row,node):
        while 'feature' in node:node=node['left'] if row[node['feature']]<=node['threshold'] else node['right']
        return node['value']
    return np.array([one(row,tree) for row in x])
def run(d,a):
    rng=np.random.default_rng(a.seed);x=rng.uniform(-2,2,(1000,3));y=5+3*(x[:,0]>0)+2*x[:,1]**2-x[:,2]+rng.normal(0,.35,1000)
    models=[(depth,fit(x[:600],y[:600],depth)) for depth in (2,4,6)]
    depth,tree=min(models,key=lambda pair:rmse(y[600:800],predict(pair[1],x[600:800])))
    return dict(selected_depth=depth,test_rmse=rmse(y[800:],predict(tree,x[800:])),baseline_rmse=rmse(y[800:],np.full(200,y[:600].mean())),tree=tree,data='1000 synthetic independent rows; 600/200/200 split'),dict(x=x,y=y)

if __name__=="__main__":main(run)
