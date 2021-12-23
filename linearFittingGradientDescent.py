

# First, we start by importing libraries:

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# Then, we generate some random data and doing the data visualization by matplotlib:

# In[16]:


xdata=np.linspace(1,10,100)
ydata=xdata*5+np.random.randn(100)*5+10
plt.scatter(xdata,ydata)


# We are trying to fit the data with a regression model here. We can speculate that this should be a straight-line (linear regression model). So, our model is y=m*x+b. 

# The choice of m and b is the most essential role here. We can consider m and b in a vector form as:
# 
# We apply the linear model, and assess the quality of a parameter by looking at the error incurred by the model prediction.

# In[17]:


def f(x,c):
    return c[0]*x+c[1]


# In[18]:


y_pred=f(xdata,[12,-17])


# In[19]:


plt.plot(xdata,ydata,'b.')
plt.plot(xdata,y_pred,'r--')


# In[20]:


def error(xdata,ydata,c):
    y_pridect=f(xdata,c)
    return (y_pred-ydata)


# In[21]:


c=[14,-10]
e=error(xdata,ydata,c)
w = (xdata[1] - xdata[0])/3
plt.bar(xdata,e,width=w)


# Next, we have to compute a loss function that assesses the quality of the model parameter which is c. The loss function is to be as follows:
# 
# Let's find the partial dervitive of Loss function relative to m and b.

# In[22]:


def Loss(c):
    e=error(xdata,ydata,c)
    L=np.sum(e**2)
    return L


# In[23]:


def Loss_grad(c):
    e=error(xdata,ydata,c)
    grad_m=np.sum(2*e*xdata)
    grad_b = np.sum(2*e)
    return np.array([grad_m,grad_b])


# In[24]:


Loss_grad(c)


# Finally we perform Gradient Descent optimization. We start by an initial guess for model parameter(c0), and step (alpha) and total number of iteration(n). 

# In[25]:


def gradientDescent(c0, alpha, n):
    c = np.array(c0)
    for i in range(n):
        c = c -( alpha * Loss_grad(c))
        L = Loss(c)
        print(i,c,L)
    return c


# In[26]:


C=gradientDescent([9,-14],1e-8,3000)


# In[27]:


type(C)


# In[28]:


plt.plot(xdata,ydata,'b.')
y_pred = f(xdata, C)
plt.plot(xdata, y_pred, 'r--')


# In[ ]:




