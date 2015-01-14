import numpy as np
import pystan
from sklearn.base import BaseEstimator

class StanModel_(pystan.StanModel):
    def __del__(self):
        """
        This method is being used carelessly in sklearn's GridSearchCV class, 
        creating and destroying copies of the estimator, which is causing the
        directory containing the compiled Stan code to be deleted.  It is 
        replaced here with an empty method to avoid this problem.  This means
        a potential proliferation of temporary directories.  
        """
        pass 

class StanEstimator(BaseEstimator):
    """
    A new sklearn estimator class derived for use with pystan.
    """
    def __init__(self,**kwargs):
        for key,value in kwargs.items():
            setattr(self,key,value)

    def set_model(self,code):
        """
        Sets and compiles a Stan model for this estimator.
        """
        self.model = StanModel_(model_code=code)
        
    def set_data(self,*args,**kwargs):
        """
        Sets the data for use with this estimator.
        Uses the 'data' keyword argument if provided, else it 
        uses the 'make_data' method.  
        """
        if 'data' not in kwargs or kwargs[data] is None:
            data = self.make_data()
        else:
            data = kwargs[data]
        self.data = data
        for key,value in data.items():
            setattr(self,key,value)

    def make_data(self,*args,**kwargs):
        """
        A model-specific method for constructing the data to be used by
        the model.  May be limited to the data passed to the Stan model's 
        fitter, or may also include other items as well.  Should return 
        a dictionary."""
        raise NotImplementedError("")

    def optimize(self,X,y):
        """
        Optimizes the estimator based on covariates X and observations y.
        """
        for key in self.data.keys():
            self.data[key] = getattr(self,key)
        self.best = self.model.optimizing(data=self.data)

    def get_params(self,deep=False):
        """
        Gets model parameters.  These are just attributes of the estimator 
        as set in __init__ and possibly in other methods.
        """
        return self.__dict__
    
    def fit(self,X,y):
        """
        Fits the estimator based on covariates X and observations y.  
        """
        self.optimize(X,y)
        for key,value in self.best.items():
            setattr(self,key,value)
        
    def transform(self,X,y=None,**fit_params):
        """
        Performs a transform step on the covariates after fitting.
        In the basic form here it just returns the covariates.  
        """
        return X

    def predict(self,X):
        """
        Generates a prediction based on X, the array of covariates.
        """  
        n_samples = X.shape[0]
        prediction = np.zeros(n_samples)
        for i in range(n_samples):
            prediction[i] = self.predict_(X,i)
            return prediction

    def predict_(self,X,i):
        """
        Generates a prediction for one sample, based on X, the array of 
        covariates and i, a point in that array (1D), or row (2D), etc.
        This must be implemented for each model.
        """
        raise NotImplementedError("")
        
    def score(self,X,y):
        """
        Generates a score for the prediction based on X, the array of
        covariates, and y, the observation.
        """
        prediction = self.predict(X)
        return self.score_(prediction,y)
        
    def score_(self,prediction,y):
        """
        Generates a score based on the prediction (from X), and the
        observation y.
        """
        raise NotImplementedError("")

    @classmethod
    def get_posterior_mean(cls,fit):
        """
        Implemented because get_posterior_mean is (was?) broken in pystan:
        https://github.com/stan-dev/pystan/issues/107
        """
        means = {}
        x = fit.extract()
        for key,value in x.items()[:-1]:
            means[key] = value.mean(axis=0)
        return means
