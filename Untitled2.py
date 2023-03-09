#!/usr/bin/env python
# coding: utf-8

# In[2]:


from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense


# In[3]:


dataset=loadtxt('diabetes.csv', delimiter=',')
x=dataset[:,0:8]
y=dataset[:,8]


# In[5]:


model=Sequential()
model.add(Dense(12,input_dim=8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


# In[6]:


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[7]:


model.fit(x,y,epochs=150,batch_size=10)


# In[16]:


model.fit(x,y,epochs=200,batch_size=10)


# In[9]:


model.fit(x,y,epochs=300,batch_size=10)


# In[15]:


_,accuracy=model.evaluate(x,y)
print ('ACCURACY :%2f=',(accuracy*100))


# In[ ]:




