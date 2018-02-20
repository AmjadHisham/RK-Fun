# http://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#tutorial-sqlsp
'''
  Maximize: f(x) = 2*x0*x1 + 2*x0 - x0**2 - 2*x1**2
  
  Subject to:    x0**3 - x1 == 0
                         x1 >= 1
'''
import numpy as np

def objective(x, sign=1.0):
    return sign*(2*x[0]*x[1] + 2*x[0] - x[0]**2 - 2*x[1]**2)

def derivative(x, sign=1.0):
    dfdx0 = sign*(-2*x[0] + 2*x[1] + 2)
    dfdx1 = sign*(2*x[0] - 4*x[1])
    return np.array([ dfdx0, dfdx1 ])

# unconstrained
result = opt.minimize(objective, [-1.0,1.0], args=(-1.0,),
                      jac=derivative, method='SLSQP', options={'disp': True})
print("unconstrained: {}".format(result.x))


cons = ({'type': 'eq',
         'fun' : lambda x: np.array([x[0]**3 - x[1]]),
         'jac' : lambda x: np.array([3.0*(x[0]**2.0), -1.0])},
        {'type': 'ineq',
         'fun' : lambda x: np.array([x[1] - 1]),
         'jac' : lambda x: np.array([0.0, 1.0])})

# constrained
result = opt.minimize(objective, [-1.0,1.0], args=(-1.0,), jac=derivative,
                      constraints=cons, method='SLSQP', options={'disp': True})

print("constrained: {}".format(result.x))


import scipy.optimize as opt
import scipy.stats as stats
import numpy as np

# Define the function to fit.
def function(x, a):
    result = a * x
    return result

# Create a noisy data set around the actual parameters
true_params = [1]
print("target parameters: {}".format(true_params))
x = train[['sqft_living']].values.flatten()
exact = function(x, *true_params)
noisy = train[['price']].values.flatten()

# Use curve_fit to estimate the function parameters from the noisy data.
initial_guess = [1]
estimated_params, err_est = opt.curve_fit(function, x, noisy, p0=initial_guess)
print("solved parameters: {}".format(estimated_params))

# err_est is an estimate of the covariance matrix of the estimates
print("covarance: {}".format(err_est.diagonal()))

import matplotlib.pylab as mpl
import matplotlib.pyplot as plt
plt.plot(x, noisy, 'ro')
plt.plot(x, function(x, *estimated_params)) 
plt.show()



from sklearn import linear_model
logitmodel = linear_model.LogisticRegression()
logitmodel.fit(train[['bathrooms']], train['view']==1)
test['view_pred']=logitmodel.predict(test[['bathrooms']])
logitmodel.intercept_ ,logitmodel.coef_

np.sum(test.view == test.view_pred)/len(test)

import scipy.optimize as opt
import scipy.stats as stats
import numpy as np

# Define the function to fit.
def function(x, b1,b2):
    result = 1./(1+np.exp(-1*(b1 + b2*x)))
    return result

# Create a noisy data set around the actual parameters
true_params = [1,1]
print("target parameters: {}".format(true_params))
x = train[['bathrooms']].values.flatten()
exact = function(x, *true_params)
noisy = (train[['view']]==1).values.flatten()

# Use curve_fit to estimate the function parameters from the noisy data.
initial_guess = [1,0.5]
estimated_params, err_est = opt.curve_fit(function, x, noisy, p0=initial_guess)
print("solved parameters: {}".format(estimated_params))

# err_est is an estimate of the covariance matrix of the estimates
print("covarance: {}".format(err_est.diagonal()))

def get_histograms(x):
    return [scipy.stats.kurtosis(x,axis=0).flatten(),
            scipy.stats.skew(x,axis=0).flatten(),
            np.mean(x,axis=0).flatten(),
            np.min(x,axis=0).flatten(),
            np.max(x,axis=0).flatten(),
            np.var(x,axis=0).flatten()]

is_iceberg_band_1 = X_full[msk,:,:,0]

# for x in get_histograms(is_iceberg_band_1):
#     plt.figure(figsize=(9,7))
#     plt.hist(x,bins=50)
#     plt.show()
    

# Define the function to fit.
def norm_1(x, height1, mu1, sigma1):
#     normal = height1 * np.exp(-1/2*((x-mu1)/sigma1)**2)
    normal = height1*stats.norm.pdf(x,mu1,sigma1)
    return normal

def gamma_1(x, height1,a,loc,scale):
#     normal = height1 * np.exp(-1/2*((x-mu1)/sigma1)**2)
    normal = height1*stats.gamma.pdf(x,a,loc=loc,scale=scale)
    return normal

def beta_1(x, height1,a,loc,scale):
#     normal = height1 * np.exp(-1/2*((x-mu1)/sigma1)**2)
    normal = height1*stats.lognorm.pdf(x,a,loc=loc,scale=scale)
    return normal

def beta_1(x, height1,a,loc,scale):
    normal = height1*stats.lognorm.pdf(x,a,loc=loc,scale=scale)
    return normal

def norm_2(x, height1, mu1, sigma1,height2, mu2, sigma2,height3, mu3, sigma3):
    normal = height1*stats.norm.pdf(x,mu1,sigma1) + height2*stats.norm.pdf(x,mu2,sigma2) \
    + height3*stats.norm.pdf(x,mu3,sigma3)
    return normal

parameters_iceberg = np.zeros((2,6,4))
                
for g in [0,1]:
    i=0
    for arr in get_histograms(X_full[msk,:,:,g]):


        vals,indx = np.histogram(arr, bins=350)

        vals = vals

        x = np.linspace(np.min(arr),np.max(arr),len(vals)) + np.diff(indx)

        min_ = np.min(x)

        x = x - min_


        initial_guess = [1.,1.,1.]
        estimated_params, err_est = opt.curve_fit(norm_1, x, vals, p0=initial_guess)

        parameters_iceberg[g,i,:] = list(estimated_params) + [min_]
        print(parameters_iceberg[g,i,:])
        i+=1
        plt.figure(figsize=(9,7))
        plt.plot(x, vals, 'ro')
        plt.plot(x, norm_1(x, *(estimated_params)),'b')
        plt.show()


        print(np.sum((vals-norm_1(x, *(estimated_params)))**2))


parameters_iceberg
