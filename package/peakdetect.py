import sys
from numpy import NaN, Inf, arange, isscalar, asarray, array

def peakdet(v, delta, x = None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    
    Returns two arrays
    
    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %      
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.
    
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    
    """
    maxtab = []
    mintab = []
       
    if x is None:
        x = arange(len(v))
    
    v = asarray(v)
    
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    
    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN
    
    lookformax = True
    
    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return array(maxtab), array(mintab)

if __name__=="__main__":
    from matplotlib.pyplot import plot, scatter, show
    import numpy as np
#    series = [0,0,0,2,0,0,0,-2,0,0,0,2,0,0,0,-2,0]
    x=np.linspace(0,np.pi*50,1000)
    series=np.exp(-0.01*x)*np.cos(x)+1.5
    maxtab, mintab = peakdet(series,1)
    plot(series,linewidth=1)
    scatter(array(maxtab)[:,0], array(maxtab)[:,1], color='blue')
    scatter(array(mintab)[:,0], array(mintab)[:,1], color='red')
    show()
    print(mintab[:,1].shape)
    
    threshold=0.5

    l=min(len(maxtab),len(mintab))
    diff=maxtab[:l,:]-mintab[:l,:] #計算峰對峰值
#        print('max',maxtab[:l,:].flatten())
#        print('min',mintab[:l,:].flatten())
    diff[:l,0]=maxtab[:l,0] #差值時間軸以max來算

    maxtab=maxtab[:l] #縮減長度
    mintab=mintab[:l] #縮減長度
    mk1=diff[:,1]>threshold #鎖定只要大角度差的數值
    mk2=diff[:,1]<130 #太大的也不要
    mk=np.logical_and(mk1,mk2) #合成mask
    trim_diff=diff[mk] 
#    print(trim_diff)
    print('max',maxtab[mk])
    print('min',mintab[mk][:,1])
    print('trim_diff',trim_diff)
