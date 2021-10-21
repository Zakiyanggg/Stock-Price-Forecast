#!/usr/bin/env python
# coding: utf-8

# In[110]:


import quandl
import array as arr
quandl.ApiConfig.api_key = '2htei_bXDXNztz2zxM2K'

# get the table for daily stock prices and,
# filter the table for selected tickers, columns within a time range
# set paginate to True because Quandl limits tables API to 10,000 rows per call

data = quandl.get_table('WIKI/PRICES', ticker = ['AAPL'], 
                        qopts = { 'columns': ['ticker', 'date', 'adj_close', 'adj_open'] }, 
                        date = { 'gte': '2013-12-31', 'lte': '2017-12-28' }, 
                        paginate=True)
data.head()

closing = data.adj_close
opening = data.adj_open
dates = data.date

list1 = []

for x in range(0, 1000):
    list1.append([closing[x],closing[x+1],closing[x+2],closing[x+3],closing[x+4],opening[x+5]])
    
for x in range(0, 1000):
    print(list1[x])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




