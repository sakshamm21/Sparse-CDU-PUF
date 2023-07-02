import numpy as np

def HT( v, k ):
    t = np.zeros_like( v )
    if k < 1:
        return t
    else:
        ind = np.argsort( abs( v ) )[ -k: ]
        t[ ind ] = v[ ind ]
        return t

def my_fit( X_trn, y_trn ):
	S = 512
	w_ls = np.array( np.linalg.lstsq( X_trn, y_trn, rcond = None )[0] )
	w_sparse = HT( w_ls, S )
	
	return w_ls

