import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
import pandas as pd
import time
import heapq

def heapsort(iterable):
    h = []
    for value in iterable:
        heapq.heappush(h, value)
    return [heapq.heappop(h) for i in range(len(h))]

def numpy_simple_regression(input_space,output):

    x,y=numpy_format_data(input_space,output)

    A=x.T.dot(x)
    b=x.T.dot(y)
    z = np.linalg.solve(A,b)
    return z

def numpy_predict(input_space,z):
    input_space,_=numpy_format_data(input_space,np.array([]))
    return input_space.dot(z)


def numpy_rsq(input_space,output,z):    
    x,y=numpy_format_data(input_space,output)
    
    return 1 - np.sum((y-x.dot(z))**2)/np.sum((y-np.mean(y))**2)

def numpy_SSE(input_space,output,z):    
    x,y=numpy_format_data(input_space,output)
    
    return np.sum((y-x.dot(z))**2)

def numpy_format_data(input_space,output):
    if type(input_space) == np.ndarray or type(input_space) == pd.core.series.Series:
        m,n =np.shape(input_space)
        x = np.column_stack((np.ones((m,1)),input_space))
    else:
        m,n =np.shape(input_space.values)
        x=np.ones((m,n+1))
        x[:,1:]=input_space.values

    if type(output) == np.ndarray:
        y=output
    else:
        y=output.values
        
    return x,y

def pandas_poly_feature(feature, degree):
    assert degree >= 1
    poly = pd.DataFrame()
    name = feature.columns
    poly[name+'_power_1'] = feature
    if degree > 1:
        for power in range(2, degree+1):
            new_name = name+'_power_' + str(power)
            poly[new_name]=feature.apply(lambda x:x**power)
    return poly



def sklearn_poly_regression(input_space,output,degree=None):
    poly_data = pandas_poly_feature(input_space, degree)
    features = poly_data.columns
    target = output.columns
    poly_data[target] = output
    
    model = linear_model.LinearRegression()
    model.fit(poly_data[features], poly_data[target])
    
    n = len(features)
    z=np.zeros(n+1).reshape((n+1,1))

    z[1:,:]=model.coef_.T
    z[:1,:]=model.intercept_
    
    return z

def numpy_poly_regression_SSE(feature,output,degree=None,z=None):
            
    input_space = pandas_poly_feature(feature, degree)
    
    y=output.values
    m,n =np.shape(input_space.values)
    x=np.ones((m,n+1))
    x[:,1:]=input_space.values

    return np.sum((y-x.dot(z))**2)

def sklearn_ridge_poly_regression(input_space,output,degree,L2_penalty):
    
    poly_data = pandas_poly_feature(input_space, degree)
    features = poly_data.columns
    target = output.columns
    poly_data[target] = output
    
    model = linear_model.Ridge(alpha=L2_penalty, normalize=True)
    model.fit(poly_data[features], poly_data[target])
    
    n = len(features)
    z=np.zeros(n+1).reshape((n+1,1))
    
    z[1:,:]=model.coef_.T
    z[:1,:]=model.intercept_

    return z

def sklearn_ridge_regression(input_space,output,L2_penalty):
    
    model = linear_model.Ridge(alpha=L2_penalty, normalize=True)
    model.fit(input_space, output)
    
    n = input_space.shape[1]
    z=np.zeros(n+1).reshape((n+1,1))
    
    z[1:,:]=model.coef_.T
    z[:1,:]=model.intercept_

    return z


def cost_function(X, y, theta):
    m = len(y) 
    cost = np.sum((X.dot(theta)-y)**2)/2/m
    return cost

def normalize_features(X):
    norms = np.linalg.norm(X, axis=0)
    return X/norms,norms

def sklearn_ridge_k_fold(k, l2_penalty, input_space, output):
    
    features = input_space.columns
    target_name = output.columns
    
    train_valid_shuffled = input_space
    train_valid_shuffled[target_name]=output
    
    n=len(train_valid_shuffled)
    #print("length of df: ",n)
    average_validation_error=0
    
    for i in range(k):
        start = (n*i)//k
        end = start + n//k + 1
        
        valid = train_valid_shuffled[start:end]
        train = train_valid_shuffled[0:start].append(train_valid_shuffled[end+1:n])
        
        #print ("iter: ",i," (start,end) = ",(start, end))                                               
        #print("valid",len(train_valid_shuffled[start:end]))
        #print("train",len(train_valid_shuffled[0:start].append(train_valid_shuffled[end+1:n])))
        #print("intercept: ",model.intercept_)
        #print("coefficients: ",model.coef_)
        #print("Validation RSS: ", model_RSS[0])        

        model = linear_model.Ridge(alpha=l2_penalty, normalize=True)
        model.fit(train[features], train[target_name])

        model_predict = model.predict(valid[features])
        model_RSS = np.sum((valid[target_name]-model_predict)**2)

        average_validation_error+=model_RSS[0]/k
    
    return average_validation_error


def sklearn_lasso_feature_selection(input_space,output,l1_penalty):
                                    
    features = input_space.columns
    model_all = linear_model.Lasso(alpha=l1_penalty, normalize=True)
    model_all.fit(input_space, output)
    
    mask = abs(model_all.coef_)>0
    return np.array(features)[mask]                               

def sklearn_lasso_regression(input_space,output,l1_penalty):
    
    
    features = input_space.columns
    
    model = linear_model.Lasso(alpha=l1_penalty, normalize=True)
    model.fit(input_space, output)
    
    n = len(features)
    z=np.zeros(n+1).reshape((n+1,1))
    
    z[1:,:]=model.coef_.T.reshape((n,1))
    z[:1,:]=model.intercept_

    return z

def sklearn_lasso_penalty_range(input_space, output, num):

    over_penalty = None
    under_penalty = None
    i=1
    for l1_penalty in np.logspace(1, 7, num=13):
        model = linear_model.Lasso(alpha=l1_penalty, normalize=True)
        model.fit(input_space, output)
        count = np.count_nonzero(model.coef_) + np.count_nonzero(model.intercept_)

        if count<num and i==1:
            print("Not enough features")
            break

        if count>=num+1:
            i+=1
            under_penalty = l1_penalty
            pass
        elif count==num:
            i+=1
            pass
        elif count<=num-1:
            over_penalty = l1_penalty
            break

    coeff_range = []
    #print(over_penalty,under_penalty)
    for l1_penalty in np.linspace(under_penalty,over_penalty,600):
        model = linear_model.Lasso(alpha=l1_penalty, normalize=True)
        model.fit(input_space, output)
        count = np.count_nonzero(model.coef_) + np.count_nonzero(model.intercept_)

        if count == num:
            coeff_range.append(l1_penalty)


    return min(coeff_range),max(coeff_range)




def vanilla_gradient_descent(input_space,output,iters,threshold,alpha,w):

    start_time = time.time()
    
    x,y=numpy_format_data(input_space,output)

    converged=False
    


    for i in range(iters):

        hypothesis = x.dot(w)
        loss = hypothesis-y
        gradient = 2*x.T.dot(loss)
        w = w - gradient*alpha
        cost = cost_function(x, y, w)
        if np.sqrt(np.sum(gradient**2))<threshold:
            converged=True
            print("execution time")
            print("--- %s seconds ---" % (time.time() - start_time))
            return w,cost,converged

        return w,cost,converged

    
    
def ridge_gradient_descent(input_space,output, alpha, l2_penalty, init_w=None, max_iterations=1000):
            
    x,y=numpy_format_data(input_space,output)
    
    if init_w==None:
        w =np.zeros(x.shape[1]).reshape((x.shape[1],1))
    else:
        w =init_w
        
    converged=False


    for i in range(max_iterations):
        
        ridge_w = w
        ridge_w[0] = 0

        hypothesis = x.dot(w)
        loss = hypothesis-y
        gradient = 2*x.T.dot(loss) + 2*l2_penalty*ridge_w
        w = w - gradient*alpha

        cost = cost_function(x, y, w)
        if np.sqrt(np.sum(gradient**2))<2.5e9:
            converged=True
            return w,alpha,cost,converged


        #print("gradient: ",np.sqrt(np.sum(gradient**2)))
        #print("alpha: ",alpha)
        #print('overshot: ', w)
    return w,alpha,cost,converged    
    
def knn_compute_distances(features_instances, features_query):
    diff = features_instances - features_query
    distances=np.sqrt(np.sum(diff**2, axis=1))
    return distances    

def numpy_knn_regression(k, predictors, target, query):
    y=target.values
    x=predictors.values
    q=query.values
    
    x , norms = normalize_features(x)
    q = q / norms

    m,n=np.shape(q)
    out = np.zeros(m)
    
    for i in range(m):

        dist = knn_compute_distances(x,q[i])
        heap=[(dist[i],i) for i in range(len(dist))]
        sorted_heap=heapsort(heap)
        k_near_houses = [i[1] for i in sorted_heap[:k]]
        out[i] = np.mean(y[k_near_houses])
        
    return out

    
    

def numpy_knn_weighted_regression(k, predictors, target, query):
    y=target.values
    x=predictors.values
    q=query.values
    
    x , norms = normalize_features(x)
    q = q / norms

    m,n=np.shape(q)
    out = np.zeros(m)
    
    for i in range(m):
        e = 1e-12
        dist = knn_compute_distances(x,q[i])
        heap=[(dist[i],i) for i in range(len(dist))]
        sorted_heap=heapsort(heap)
        k_near_houses = [i[1] for i in sorted_heap[:k]]
        inv_local_dist = 1./(np.array([i[0] for i in sorted_heap[:k]])+e)
        n=len(inv_local_dist)
        if k==1:
            out[i] = np.sum(y[k_near_houses])
        else:
            inv_dist = 1./(np.array(dist[k_near_houses]).reshape((1,n))+e)
            local_weight = inv_dist/np.sum(inv_local_dist)
            out[i] = local_weight.dot(y[k_near_houses])
        
    return out




def numpy_gaussian_kernel_regression(k,b,predictors, target, query):
    
    def compute_abs_distances(features_instances, features_query):
        diff = features_instances - features_query
        distances=np.sqrt(np.sum(np.abs(diff), axis=1))
        return distances
    
    y=target.values
    x=predictors.values
    q=query.values
    e = 1e-12
    x , norms = normalize_features(x)
    q = q / norms

    m,n=np.shape(q)
    out = np.zeros(m)
    
    for i in range(m):
        dist = compute_abs_distances(x,q[i])
        msk = dist<b
        if np.sum(msk)<k:
            heap=[(dist[i],i) for i in range(len(dist))]
            sorted_heap=heapsort(heap)
            k_near_houses = [i[1] for i in sorted_heap[:k]]
            msk = k_near_houses
            
        cqi = np.exp(-np.array([dist[msk]])/b)

        yi = y[msk]
        kernel =  cqi.dot(yi)/np.sum(cqi)
        out[i] = kernel

#             print(cqi)
#             print()
#             print(yi)
#             print()
#             print(kernel)
#             print()
        
    return out


def adadelta_gradient_descent(x, y, iters, alpha,w):
    
    def cost_function(X, y, theta):
        m = len(y) 
        J = np.sum((X.dot(theta)-y)**2)/2/m
        return J

    v_history = 0

    for i in range(iters):
        
        change=0
        converged=False
        done=True
        hypothesis = x.dot(w)
        loss = hypothesis-y
        gradient = x.T.dot(loss)
        if i==0:
            v = gradient**2
            rms_v = np.sqrt(v+0.0000000001)
            update= -1.0*(np.sqrt(0.0000001)/rms_v)*gradient
            parameter_history = update**2
        
        else:
            v = (v_history)*0.9 + 0.1*gradient**2
            
            rms_v = np.sqrt(v+0.0000000001)
            update= -1.0*(np.sqrt(parameter_history+0.0000001)/rms_v)*gradient

        
        parameter_history = parameter_history*0.9 + 0.1*update**2
        
        w = w + update*alpha
        
        v_history = v
        
    cost = cost_function(x, y, w)

    return w,alpha,cost,gradient

