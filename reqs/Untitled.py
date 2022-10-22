#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


# Voila Web App

## A website built out of a Jupyter notebook using Voila


# In[2]:


# import numpy as np 
import matplotlib.pyplot as plt
import ipywidgets as widgets
import pandas as pd
import seaborn as sns
plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
import numpy as np
warnings.filterwarnings('ignore', category=DeprecationWarning)
from warnings import filterwarnings
filterwarnings('ignore')


# In[3]:


#load data
companies = pd.read_csv("kc_house_data.csv", sep = ',')
#drop id column
companies = companies.iloc[: , 1:]
companies.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[4]:


#heatmap
sns.heatmap(companies.corr(), cmap='RdBu')


# In[5]:


with sns.plotting_context("notebook",font_scale=2.5):
     g = sns.pairplot(companies[['sqft_lot','sqft_above','price','sqft_living','bedrooms']], 
                  hue='bedrooms', palette='tab20',size=5)
g.set(xticklabels=[]);


# In[6]:


plt.figure(figsize=(15,10))
plt.tight_layout()
sns.histplot(companies['price'], legend=True, stat = "count")


# In[7]:


#assign x and y
temp = companies.drop(['date','sqft_living15','sqft_lot15', 'price', 'lat', 'long', 'yr_renovated', 'view'], axis=1)
x = temp.values
y = companies.iloc[:,1].values


# In[8]:


temp.head()


# In[9]:


temp.describe()


# In[ ]:





# In[10]:


from sklearn.model_selection import train_test_split, GridSearchCV
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[ ]:





# In[11]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
ml=LinearRegression(fit_intercept=False)
ml.fit(x_train,y_train)
print(ml)


# In[ ]:





# In[12]:


print(len(y_train))
print(len(x_train))
print(len(x_test))
print(len(y_test))


# In[13]:


y_pred1=ml.predict(x_test)
y_pred=y_pred1


# In[ ]:





# In[ ]:





# In[14]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print("r squared score of the data: " + str(r2))


# In[ ]:





# In[15]:


import matplotlib.pyplot as plt
plt.scatter(y_test,y_pred, s=40)
plt.xlabel("Actual")
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')


# In[16]:


isThere = []
index = -1
for item in y_test:
    index = index + 1
    if item <= y_pred[index] + 250000 and item >= y_pred[index] - 250000:
        isThere.append('TRUE')
    else:
        isThere.append('FALSE')


# In[17]:


count =0
total = len(isThere)
for item in isThere:
    if item == 'TRUE':
        count = count + 1
ratio = count/total
print(count)
print(total)
print (ratio)


# In[ ]:





# In[18]:


pred_y_df=pd.DataFrame({'Actual Value':y_test.flatten(),'Predicted value':y_pred.flatten(), 'Difference':y_test-y_pred, 'inRange?':isThere})


# In[19]:


df1 = pred_y_df.head(35)
df1.plot(kind='bar',figsize=(20,15))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[20]:


print("number of bedrooms")
bedrooms = widgets.IntSlider(min = 0, max = 33)
display(bedrooms)


# In[21]:


print("number of bathrooms")
bathrooms = widgets.IntSlider(min = 0, max = 8)
display(bathrooms)


# In[22]:


print("Square Feet")
square_feet_living = widgets.IntSlider(min = 300, max = 14000)
display(square_feet_living)


# In[23]:


print("Square Feet (lot size)")
square_feet_lot = widgets.IntSlider(min = 520, max = 10000)
display(square_feet_lot)


# In[24]:


print("floors")
floors = widgets.IntSlider(min = 1, max = 3)
display(floors)


# In[25]:


wf = widgets.Checkbox(
    value=False,
    description='waterfront property?',
    disabled=False,
    indent=False
)
display(wf)


# In[26]:


print("condition (worst to best)")
condition = widgets.IntSlider(min = 1, max = 5)
display(condition)


# In[27]:


print("Overall Grade")
grade = widgets.IntSlider(min = 1, max = 13)
display(grade)


# In[28]:


print("Square feet above")
square_feet_above = widgets.IntSlider(min = 300, max = 9500)
display(square_feet_above)


# In[29]:


print("Square feet below (put 0 if there is no basement)")
square_feet_below = widgets.IntSlider(min = 0, max = 2000)
display(square_feet_below)


# In[30]:


print("Year built")
Year_built = widgets.IntSlider(min = 1900, max = 2015)
display(Year_built)


# In[31]:


print("Zip Code")
Zip_Code = widgets.IntSlider(min = 98001, max = 98200)
display(Zip_Code)


# In[32]:


checkbox = 0
from IPython.display import display
button = widgets.Button(description="Calculate")
output = widgets.Output()
prediction = 0

display(button, output)

def on_button_clicked(b):
    with output:
        sfl =square_feet_living.value
        sfa = square_feet_above.value
        sflot =square_feet_lot.value
        sfb =square_feet_below.value
       
        if wf.value == False:
            checkbox = 0
        else:
            checkbox = 1
        prediction = np.array([bedrooms.value, bathrooms.value, square_feet_living.value, square_feet_lot.value, float(floors.value), checkbox, 
                              condition.value, grade.value, square_feet_above.value, square_feet_below.value, Year_built.value, Zip_Code.value])
        prediction = prediction.reshape(-1, 12)
        predict = ml.predict(prediction)
        if  sfl <= sfa + sfb:
            print("square feet living must be more than sqare feet above and square feet basement")
        elif predict < 250000:
            print('hmmm the values you entered produced a very small house value, please make sure you have your numbers entered correctly')
        else:
            number = "{:,}".format(int(predict))
            low = "{:,}".format(int(predict) - 250000)
            high=  "{:,}".format(int(predict) + 250000)
            print("House predicted Value: $" + number)
            print("This house has a 86 percent chance of being valued somewhere between $" + low
            + " and $" +high)
                  
button.on_click(on_button_clicked)


# In[3]:


import session_info
session_info.show()


# In[ ]:




