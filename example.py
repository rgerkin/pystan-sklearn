import numpy as np
from scipy.stats import norm
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import ShuffleSplit
from __init__ import StanEstimator

#############################################################
# All of this from the eight schools example.  
schools_code = """
    data {
        int<lower=0> J; // number of schools
        real y[J]; // estimated treatment effects
        real<lower=0> sigma[J]; // s.e. of effect estimates
    }
    parameters {
        real mu;
        real<lower=0> tau;
        real eta[J];
    }
    transformed parameters {
        real theta[J];
        for (j in 1:J)
        theta[j] <- mu + tau * eta[j];
    }
    model {
        eta ~ normal(0, 1);
        y ~ normal(theta, sigma);
}
"""

schools_dat = {'J': 8,
               'y': [28,  8, -3,  7, -1,  1, 18, 12],
               'sigma': [15, 10, 16, 11,  9, 11, 10, 18]}
#############################################################

# First we have to make an estimator specific to our model.  
# For now, I don't have a good way of automatically implementing this
# in a general way based on the model code.  
class EightSchoolsEstimator(StanEstimator):
    # Implement a make_data method for the estimator.  
    # This tells the sklearn estimator what things to pass along
    # as data to the Stan model.  
    # This is trivial here but can be more complex for larger models.  
    def make_data(self,search_data=None):
        data = schools_dat
        if search_data:
            data.update({key:value[0] for key,value in search_data.items()})
        return data

    # Implement a predict_ method for the estimator.  
    # This tells the sklearn estimator how to make a prediction for one sample.  
    # This is based on the prediction for the mean theta above.   
    def predict_(self,X,j):
        print(X,j)
        theta_j = self.mu + self.tau * self.eta[j];
        return theta_j
    
    # Implement a score_ method for the estimator.  
    # This tells the sklearn estimator how to score one observed sample against
    # the prediction from the model.  
    # It is based on the fitted values of theta and sigma.     
    def score_(self,prediction,y):
        likelihoods = norm.pdf(y,prediction,self.sigma)
        return np.log(likelihoods).sum()
    
# Initialize StanEstimator instance.  
estimator = EightSchoolsEstimator() 
# Compile the model code.
estimator.set_model(schools_code)   

# Search over these parameter values.  
search_data = {'mu':[0.3,1.0,3.0]} 
# Create a data dictionary for use with the estimator.  
# Note that this 'data' means different things in sklearn and Stan.  
data = estimator.make_data(search_data=search_data) 
# Set the data (set estimator attributes).  
estimator.set_data(data) 

# Fraction of data held out for testing.  
test_size = 0.1 
# A cross-validation class from sklearn.  
# Use the sample size variable from the Stan code here (e.g. "J").  
cv = ShuffleSplit(data['J'], test_size=test_size) 
# A grid search class over parameters from sklearn.
grid = GridSearchCV(estimator, search_data, cv=cv)   
    
# Set the y data.  
# Use the observed effect from the Stan code here (e.g. "y").  
y = data['y']
# Set the X data, i.e. the covariates.  
# In this example there is no X data so we just use an array of ones.   
X = np.ones((len(y),1))
#vstack((data['subject_ids'],data['test_ids'])).transpose()

# Fit the model over the parameter grid.  
grid.fit(X,y)

# Print the parameter values with the best scores (best predictive accuracy).  
print(grid.best_params_)        
