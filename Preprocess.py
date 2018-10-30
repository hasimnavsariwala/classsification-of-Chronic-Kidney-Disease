
# coding: utf-8

# In[ ]:



import pandas as pd



# In[ ]:



df = pd.read_csv('raw_data.csv')



# In[ ]:



df.head()



# In[ ]:



numeric = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']



# In[ ]:



def convert_numeric(df, attr):
    
X = df[df[attr] != '?'][attr].astype(float)
    
return X.mean()



# In[ ]:



def convert_normal(df,  attr):
    
X = df[df[attr] != '?'][attr]
    
return X.mode()[0]



# In[ ]:



for i in df.columns:
    
	#print(i, ": ")
    
	if i in numeric:
        
		ret = convert_numeric(df, i)
        
		df[i] = df[i].replace(['?'], ret)
        
		#print(ret)
    
	else:
        
		ret = convert_normal(df,i)
        
		df[i] = df[i].replace(['?'], ret)
        
		#print(ret)



# In[ ]:



df.to_csv('preprocessed_data.csv')

