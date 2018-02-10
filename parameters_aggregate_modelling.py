class Grouper():
    
    def __init__(self,GROUP,BY,FUNC):
    
        self.by_ = BY
        self.group_ = GROUP
        self.func_ = FUNC
    
    

class PandasGroupByHelper():

    def __init__(self,DF):
        
        self.df = DF
        self.N = len(self.df)
        
    def group_by(self,GROUP,BY,FUNC):
        
        self.by_ = BY
        self.group_ = GROUP
        self.func_ = FUNC
        
        self.grouper()

        
    def grouper(self):
        aggregate = [x for x in self.group_ if x not in self.by_]
        self.groupby_df = self.df[self.group_].groupby(self.by_).apply(lambda x: self.func_(x[aggregate])).reset_index()
        
    def split_(self,PERC_TRAIN=0.3):
        percent = PERC_TRAIN
        self.unique_ = self.df[self.by_[0]].unique()
        unique_df = pd.DataFrame(self.unique_)
        unique_df.columns = self.by_
        
        mask = unique_df[self.by_][int(len(self.unique_)*percent):]
        self.train = self.df.merge(mask,on=self.by_,how='inner')
        self.groupby_train = self.groupby_df.merge(mask,on=self.by_,how='inner')
        
        
        mask = unique_df[self.by_][int(len(self.unique_)*percent):]
        self.valid = self.df.merge(mask,on=self.by_,how='inner')
        self.groupby_valid = self.groupby_df.merge(mask,on=self.by_,how='inner')
        
        
    def model_weights(self,INPUT,OUTPUT,IS_CAT):
        
        assert len(OUTPUT) == 1
        
        N = len(INPUT)
        self.features_ = []
        for i in range(N):
            
            if IS_CAT[i] == 1:
                dummies = pd.get_dummies(self.df[INPUT[i]], prefix=INPUT[i]).iloc[:, 1:]
                self.df = pd.concat([self.df, dummies], axis=1)
                self.features_ += [x for x in self.df.columns if INPUT[i]+'_' in x ] 
            else:
                self.features_ += INPUT[i]

        weights = np.zeros((len(self.unique_),len(self.features_)+1))
        
        i=0
        for unique in self.unique_:
        
            tmp = self.df[self.df[self.by_[0]] == unique][OUTPUT+self.features_]
    
            y = tmp[OUTPUT[0]]
            X = tmp[self.features_]

#             print(y,X)
            model = linear_model.LinearRegression()
            model.fit(X, y)
            line_y = model.predict(X)

#             scikit_learn_SSE = np.sum((y-line_y)**2)
#             scikit_learn_coeff = model.coef_

#             plt.plot(line_y, y, 'go')
#             plt.show(
#             )
                          
            true_w = ([model.intercept_] + list(model.coef_))
#             if len(true_w)!=7:

            weights[i,:] = true_w

            i+=1
            
        
        print(weights.shape,self.features_)
        self.weights_df = pd.DataFrame()
        self.weights_df[['intercept']+self.features_] = pd.DataFrame(weights)
        self.weights_df[self.by_[0]] = self.unique_
        
        self.groupby_train = self.groupby_train.merge(weights_df,on=['Store'],how='inner')
        self.groupby_valid = self.groupby_valid.merge(weights_df,on=['Store'],how='inner')
        
full.Store = full.Store.apply(int)
full = full[full.Store<500]
full.Value = full.Value.apply(int)
full.Promo = full.Promo.apply(int)
full.Open = full.Open.apply(int)
full.DayOfWeek = full.DayOfWeek.apply(int)
full.Customers = full.Customers.apply(int)
full.Value = full.Value.apply(int)
full.fillna(0)

def test(x):
#     print(x)
    return np.sum(x)

helper = PandasGroupByHelper(full)

helper.group_by(['Customers','Store','Value'],['Store'],test)

helper.split_(PERC_TRAIN=0.3)

helper.df.head()

helper.model_weights(['DayOfWeek'],['Value'],IS_CAT=[1])

helper.weights_df.head()

model = linear_model.LinearRegression()
model.fit(helper.df[helper.features_], helper.df['Value'])
true_w = ([model.intercept_] + list(model.coef_))
true_w

X = helper.groupby_train[helper.features_]
y = (helper.groupby_train.Customers > np.mean(helper.groupby_df.Customers)).astype(np.int)

model = linear_model.LogisticRegression()
model.fit(X, y)
line_y = model.predict(X)

np.sum(y==line_y)

X = helper.groupby_train[['Value']]
y = (helper.groupby_train.Customers > np.mean(helper.groupby_df.Customers)).astype(np.int)

model = linear_model.LogisticRegression()
model.fit(X, y)
line_y = model.predict(X)

np.sum(y==line_y)