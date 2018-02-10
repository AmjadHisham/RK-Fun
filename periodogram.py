class Periodogram():
    def __init__(self, DATA):

        self.N = len(DATA)
        self.Y = DATA
        self.index = np.arange(1,self.N+1)
        self.fft()

    def fft(self):
        self.Yf = np.fft.fft(signal.detrend(self.Y))
        self.freq= np.fft.fftfreq(self.N)
        self.F = self.freq[:self.N//2]
        self.MAG = self.Yf[:self.N//2]
        self.MAG = 2.0/self.N * np.abs(self.MAG)
        
    def get_seasonality(self,MAX_PERIOD=30,PERIOD_N=2):
        derivative = np.convolve(p.MAG,[-1,0,1],'same')
        msk = derivative<0
        f_msk = 1/p.F > MAX_PERIOD
        derivative[msk] = 0
        derivative[f_msk] = 0
        
        heap = np.column_stack((derivative[1:],p.F[:-1]))
        
        sorted_heap = Regression.heapsort([(m,f) for m,f in heap])
        
        self.S = []
        for (m,f) in reversed(sorted_heap):
            period = round(1/f,0)
            if (period not in self.S):
                self.S.append(period)
                
            if (len(self.S)==PERIOD_N):
                break
        
    def make_predictors(self):
        
        count=0
        for period in self.S:
            sig = (self.index%period)
            for i in range(1,int(period)):
                
                if count==0:
                    season = (sig==i).astype(np.int).reshape(self.N,1)
                else:
                    tmp = (sig==i).astype(np.int).reshape(self.N,1)
                    season = np.column_stack((season,tmp))  
                    
                count+=1
        return season
                
        
        
p = Periodogram(full[['Pageviews']].values.flatten())

p.fft()

p.get_seasonality(MAX_PERIOD=200,PERIOD_N=2)
p.S
season = p.make_predictors()
S_train = season[:600,:]
S_valid = season[600:,:]

plt.plot(p.F,p.MAG)
plt.show()



heap = []

T_train = np.column_stack((train[['index']],train[['index']]**2))
T_valid = np.column_stack((valid[['index']],valid[['index']]**2))


S_train = season[:600,:]
S_valid = season[600:,:]


P_train_df = pd.DataFrame(np.column_stack((S_train,T_train)))
P_valid_df = pd.DataFrame(np.column_stack((S_valid,T_valid)))  

lasso_heap=[]

for l1_penalty in np.logspace(0.01, 7, num=30):
    z = Regression.sklearn_lasso_regression(P_train_df,train[['Pageviews']],l1_penalty)
    SSE = Regression.numpy_SSE(P_valid_df,valid[['Pageviews']],z)
    lasso_heap.append((SSE,l1_penalty))
penalty = Regression.heapsort(lasso_heap)[0][1]
mask = Regression.sklearn_lasso_feature_selection(P_train_df,train[['Pageviews']],penalty)

z = Regression.numpy_simple_regression(P_train_df.values,train[['Pageviews']])

plt.plot(train[['Pageviews']].values.flatten())
plt.plot(Regression.numpy_predict(P_train_df.values,z).flatten())
plt.show()

SSE = Regression.numpy_SSE(P_train_df.values,train[['Pageviews']],z)
SST = np.sum((train[['Pageviews']].values - np.mean(train[['Pageviews']].values))**2)

SSE = Regression.numpy_SSE(P_valid_df.values,valid[['Pageviews']],z)
SST = np.sum((valid[['Pageviews']].values - np.mean(valid[['Pageviews']].values))**2)
heap.append((1-SSE/SST,power))

plt.plot(valid[['Pageviews']].values.flatten())
plt.plot(Regression.numpy_predict(P_valid_df.values,z).flatten())
plt.plot(np.repeat(np.mean(valid[['Pageviews']].values.flatten()),len(valid)))
plt.show()

Regression.heapsort(heap),z