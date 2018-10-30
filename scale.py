#scaling
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('unscaled_data.csv')
df.head()



# In[ ]:



numeric = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']



# In[ ]:




def scale_numeric(df, attr):
	X = df[attr].astype(float)
	scaler = MinMaxScaler(feature_range=(0,1))
	scaled_X= scaler.fit_transform(X)
	return scaled_X



def scale_word(df, attr):
	X = df[attr].astype(str)
	scaled_X = X
	return scaled_X


# In[ ]:



for i in df.columns:
    
	#print(i, ": ")
    
	if i in numeric:
        
		ret = scale_numeric(df,i)

		df[i]= df[i].replace(i,ret)    
	else:
        
		ret = scale_word(df,i)
		df[i] = df[i]


np.set_printoptions(precision=3)
print(pd.DataFrame(scaled_X))

df = pd.DataFrame(scaled_X)
df.to_csv("scaled_data.csv")