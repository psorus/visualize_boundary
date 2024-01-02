import numpy as np


def raster(dim=2,xmin=-1,xmax=1,n=10):
    """
    Returns an array of shape (n**dim,dim) with all equally spaced points between xmin and xmax in dim dimensions
    """
    #if xmin/xmax is a number, repeat it dim times
    if not hasattr(xmin,'__len__'):
        xmin = [xmin]*dim
    if not hasattr(xmax,'__len__'):
        xmax = [xmax]*dim
    return np.array(np.meshgrid(*[np.linspace(xmin[i],xmax[i],n) for i in range(dim)])).reshape(dim,-1).T

def multiraster(xmin,xmax,dim=2,n=10):
    """
    Returns an array of shape (n**dim,dim) with all equally spaced points between xmin and xmax in dim dimensions
    now assumes xmin, xmax to be a list, returns rasteration for each
    """
    ret=[]
    for xmi,xma in zip(xmin,xmax):
        ret.append(raster(dim,xmi,xma,n))
    return np.array(ret)



if __name__ == "__main__":
    q=raster(3,-1,1,10)
    print(q)
    print(q.shape)


