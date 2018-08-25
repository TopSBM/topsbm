"""
This is the main file for implementing the hSBM for text mining.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances


class hSBMTransformer(BaseEstimator, TransformerMixin):
    """ An example transformer that returns the element-wise square root..

    Parameters
    ----------
    demo_param : str, optional
        A parameter used for demonstation of how to pass and store paramters.

    Attributes
    ----------
    input_shape : tuple
        The shape the data passed to :meth:`fit`
    """
    def __init__(self, demo_param='demo'):
        self.demo_param = demo_param
        
        ## VJ - Check if all this init are needed
        ## Also need to add the relevant hyper-parameters.
        
        self.g = None ## network

        self.words = [] ## list of word nodes
        self.documents = [] ## list of document nodes

        self.state = None ## inference state from graphtool
        self.groups = {} ## results of group membership from inference
        self.mdl = np.nan ## minimum description length of inferred state
        self.L = np.nan ## number of levels in hierarchy

    def __make_graph(self,documents = None, counts=True):
        print('make_graph is private, need to be called from fit')
          
    
    def fit(self, X, y=None):
        """Fit the hSBM topic model
    
        Constructs a graph representation of X, infers clustering, and reports
        the cluster probability for each sample in X.
    
        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)
            Word frequencies for each document, represented as  non-negative
            integers.
        y : ignored
    
        Returns
        -------
        Xt : ndarray of shape (n_samples, n_components)
        """
        X = check_array(X)
        self.input_shape_ = X.shape
        
        print("Hello I am in hSBMTransformer:fit()")
        
    
        # Return the transformer OR the matrix??
        return self
    
    
    
    

    def transform(self, X):
        """ A reference implementation of a transform function.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        X_transformed : array of int of shape = [n_samples, n_features]
            The array containing the element-wise square roots of the values
            in `X`
        """
        # Check is fit had been called
        check_is_fitted(self, ['input_shape_'])

        # Input validation
        X = check_array(X)

        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape != self.input_shape_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
            
        print("Hello I am in hSBMTransformer:transform()")
        return np.sqrt(X)
