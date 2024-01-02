import numpy as np

from tqdm import tqdm

#epsilon=(max(xmax)-min(xmin))/n**steps
#epsilon=0.001*epsilon
epsilon=1e-7

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

#dim=2
#n=50
#n0=20
#n=10
#xmin=[-3.0 for _ in range(dim)]
#xmax=[3.0 for _ in range(dim)]

#border=1.0*2

#recursion_steps=3-1



def multistep(func,border,dim,n,xmin,xmax):
    xmin=np.array(xmin)
    xmax=np.array(xmax)


    
    pointss=multiraster(dim=dim, n=n, xmin=xmin, xmax=xmax)
    shap=list(pointss.shape[:-1])
    pointss=pointss.reshape(-1,dim)
    
    valuess=func(pointss)

    valuess=valuess.reshape(shap)
    pointss=pointss.reshape(shap+[dim])

    def substep(points,values,xmin,xmax,n=n):

        width=(xmax-xmin)
        outervolume=np.prod(width)
        subvolume=outervolume/(n-1)**dim
        
        
        local=np.reshape(points,[n]*dim+[dim])
        
        
        condition=np.reshape(values,[n]*dim)<border
        
    
        def one_roller(arr,pattern):
            arr=np.copy(arr)
            assert len(pattern)==dim
            for i,p in enumerate(pattern):
                if p:
                    arr=np.roll(arr, 1, axis=i)
            return arr
        def all_patterns(dim=dim):
            if dim==0:
                yield []
            else:
                for i in range(2):
                    for p in all_patterns(dim=dim-1):
                        yield [i]+p
    
        def check_pattern(arr,pattern):
            arrr2=one_roller(arr,pattern)
            return np.logical_xor(arr, arrr2)
        def check_all_pattern(arr,dim=dim):
            return np.logical_or.reduce([check_pattern(arr,p) for p in all_patterns(dim=dim) if np.sum(p)>0])
    
        globalpos=one_roller(local,[1]*dim)
        globalroll=check_all_pattern(condition)#np.logical_xor(globalroll, condition)
        altpos=globalpos+width/(n-1)
    
        def isvalid(poi):
            return np.all(poi>=xmin-epsilon) and np.all(poi<=xmax+epsilon)
    
        totake=[]
        internal=0
        surface=0
        valids=0
        areas=[]
        for c,a,b,o in zip(globalroll.reshape(-1),globalpos.reshape(-1,dim),altpos.reshape(-1,dim),condition.reshape(-1)):
            if not isvalid(a) or not isvalid(b):
                continue
            valids+=1
            if c:
                totake.append(np.stack((a,b),axis=-1).T)
                surface+=1
            elif o:
                areas.append(np.stack((a,b),axis=-1).T)
                internal+=1
    
        
    
        #exit()
    
        totake=np.array(totake)
        areas=np.array(areas)
        volume=internal*subvolume
        surface=surface/valids#lets to this differently#*subvolume/width[0]#bodge, assumes that all xmin and xmax are the same

        surface_factor=subvolume**((dim-1)/dim)

        return volume,surface,totake,areas,surface_factor

    volumes,surfaces,totakes,areas,surface_factor=[],[],[],[],None
    for points,values,xmi,xma in zip(pointss,valuess,xmin,xmax):
        volume,surface,totake,area,surface_factor=substep(points,values,xmi,xma,n=n)
        volumes.append(volume)
        surfaces.append(surface)
        for take in totake:
            totakes.append(take)
        for ar in area:
            areas.append(ar)
        #totakes.append(totake)
        if surface_factor is None:
            surface_factor=surface_factor
    totakes=np.array(totakes)
    areas=np.array(areas)
    #print(totakes.shape)
    #exit()
    

    return volumes, surfaces, totakes, areas, surface_factor

def concatplus(a,b):
    if len(a)==0:return b
    if len(b)==0:return a
    return np.concatenate((a,b),axis=0)

def rsurf(func,border,n0=20,n=10,xmin=0,xmax=0,dim=3,steps=2):
    xmin=[xmin for _ in range(dim)]
    xmax=[xmax for _ in range(dim)]
    
    
    vol,surf,pos,area,surface_factor=multistep(func,border,n=n0,xmin=[xmin],xmax=[xmax],dim=dim)
    vol=vol[0]
    surf=surf[0]
    def recursion(vol,surf,pos,area):
        newvol,newsurf,newpos=vol,0,[]
        xmins,xmaxs=[],[]
        for xmin,xmax in pos:
            xmins.append(xmin)
            xmaxs.append(xmax)
        itervol,itersurf,iterpos,iterarea,surface_factor=multistep(func,border,n=n,xmin=xmins,xmax=xmaxs,dim=dim)
        newvol=sum(itervol)
        newsurf=sum(itersurf)
        for zw in iterpos:
            newpos.append(zw)
        newsurf=newsurf*surf/len(pos)
        area=concatplus(area,iterarea)
        return vol+newvol,newsurf,newpos,area,surface_factor
    
    for i in range(steps-1):
        vol,surf,pos,area,surface_factor=recursion(vol,surf,pos,area)
    
    mesh_surface=np.array(pos)
    mesh_volume=np.array(area)
    pos=np.array(pos)
    ds=[pos[:,1,i]-pos[:,0,i] for i in range(dim)]
    ds=np.array(ds)
    #print(ds.shape)
    #exit()
    vol+=np.sum(np.prod(ds,axis=0))/2
   
    return vol,surf,mesh_surface,mesh_volume
    

def plot(mesh_surface,mesh_volume,color="red",fill=True,label=None):
    import matplotlib.pyplot as plt
    
    
    import matplotlib.patches as patches
    
    #add rectangle
    if fill:
        for (lowx,lowy),(upx,upy) in mesh_volume:
            plt.gca().add_patch(patches.Rectangle((lowx,lowy),upx-lowx,upy-lowy,linewidth=1,edgecolor='none',facecolor=color,alpha=0.1))
    for (lowx,lowy),(upx,upy) in mesh_surface:
        plt.gca().add_patch(patches.Rectangle((lowx,lowy),upx-lowx,upy-lowy,linewidth=1,edgecolor=color,facecolor=color,alpha=0.5,label=label))
        label=None


if __name__=="__main__":

    #def rsurf(func,border,n0=20,n=10,xmin=0,xmax=0,dim=3,steps=2):

    from time import time

    for steps in range(1,7):
        t0=time()
        func=lambda x:np.sqrt(np.sum(np.square(x),axis=-1))
        volume,_,_,_=rsurf(func,1.0,n0=20,n=10,xmin=-1,xmax=1,dim=2,steps=steps)
        t1=time()
        print("Using steps:",steps)
        print("Found Volume:",volume)
        print("Required time:",t1-t0)
        print("Error:",round(100*(np.pi-volume)/np.pi,8),"%")











