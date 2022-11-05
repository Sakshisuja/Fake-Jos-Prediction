#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Install wordcloud

pip install wordcloud


# In[2]:


#Install spacy
pip install spacy


# In[4]:


#Importing all needed libraries

import re
import string
import numpy as np
import pandas as pd


import random
import matplotlib.pyplot as plt
import seaborn as sns #Seaborn is a Python data visualization library based on matplotlib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin #ransformers are scikit-learn estimators which implement a transform method
from sklearn.metrics import accuracy_score,plot_confusion_matrix,classification_report,confusion_matrix
from wordcloud import WordCloud
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from sklearn.svm import SVC


# In[5]:


df=pd.read_csv('C:/Sakshi/fake_job_postings.csv')
df.head()


# In[6]:


df.shape


# In[7]:


df.isnull().sum()


# In[6]:


#Due to many null values drop the columns which have them
columns=['job_id','telecommuting','has_company_logo','has_questions','salary_range','employment_type']
for co in columns:
    del df[co]
    


# In[7]:


df.head()


# In[8]:


#To handle missing rest values fill NaN with blank
df.fillna('',inplace=True)


# In[11]:


#Lets visulize no of fraudulent and real job posts
plt.figure(figsize=(15,6))
sns.countplot(y='fraudulent',data=df)
plt.show()


# In[12]:


#Get exact count of fraudulent and non fraudulent
df.groupby('fraudulent')['fraudulent'].count()


# In[13]:


#Visualize no of jobs and experience they demanded.required experience is variable we have in our
#a datset and we need to delete all blanks
exp=dict(df.required_experience.value_counts())
del exp['']
print(exp)


# In[14]:


plt.figure(figsize=(10,5))
sns.set_theme(style='whitegrid')
plt.bar(exp.keys(),exp.values())
plt.title("no of jobs with experiences",size=20)
plt.xlabel("Experience",size=10)
plt.ylabel("No of jobs",size=10)
plt.xticks(rotation=30)
plt.show()


# In[15]:


def split(location):
    l=location.split(',')
    return l[0]
df['country']=df.location.apply(split)
df.head()


# In[16]:


#Now visualize some top countries and no of jobs they have
countr=dict(df.country.value_counts()[:14])
del countr['']
print(countr)


# In[17]:


#Now visualize graph for education level and no of jobs posted for that education level
plt.figure(figsize=(8,6))
plt.bar(countr.keys(),countr.values())
plt.title("contrywise job posting",size=20)
plt.xlabel("countries",size=10)
plt.ylabel("No of jobs",size=10)


# In[18]:


#create key value pair and visualize some top records
edu=dict(df.required_education.value_counts()[:7])
del edu['']
edu


# In[17]:


plt.figure(figsize=(15,6))
plt.title('Job postings Based on Education',size=20)
plt.bar(edu.keys(),edu.values())
plt.ylabel('no. of jobs',size=10)
plt.xlabel('Education',size=10)


# In[19]:


#Print the titles of jobs which are real
#Similarly Print the titles of jobs which are fraudulent

print(df[df.fraudulent==0].title.value_counts()[:10])


# In[20]:


print(df[df.fraudulent==1].title.value_counts()[:10])


# In[21]:


#Combine all columns in dataset in column that is text .will be remaining only two cols one is whether job posted is 
#fraudulent or not and other is text column delete the rest of columns

df['text']=df['title']+' '+df['company_profile']+' '+df['description']+' '+df['requirements']+' '+df['benefits']
del df['title']
del df['location']
del df['department']
del df['company_profile']
del df['description']
del df['requirements']
del df['benefits']
del df['required_experience']
del df['required_education']
del df['industry']
del df['function']
del df['country']


# In[22]:


df.head()


# In[23]:



fraud_jobs_text=df[df.fraudulent==1].text
real_jobs_text=df[df.fraudulent==0].text


# In[24]:


#Create a word cloud based on text for fake and realjobs
STOPWORDS=sspacy.lang.en.stop_words.STOP_WORDS
plt.figure(figsize=(16,14))
wc=WordCloud(min_font_size=3,max_words=3000,width=1600,height=800,stopwords=STOPWORDS).generate(str(" ".join(fraud_jobs_text)))
plt.imshow(wc,interpolation='bilinear')


# In[25]:


STOPWORDS=spacy.lang.en.stop_words.STOP_WORDS
plt.figure(figsize=(16,14))
wc=WordCloud(min_font_size=3,max_words=3000,width=1600,height=800,stopwords=STOPWORDS).generate(str(" ".join(real_jobs_text)))
plt.imshow(wc,interpolation='bilinear')


# In[26]:


#To work with english words in nlp we need to install some other packages.
get_ipython().system('pip install spacy && python -m spacy download en')


# In[27]:


punctuations= string.punctuation
nlp= spacy.load("en_core_web_sm")

stop_words=spacy.lang.en.stop_words.STOP_WORDS
parser=English()

def spacy_tokenizer(sentence):
    mytokens=parser(sentence)
    mytokens=[word.lema_lower().strip() if word.lema_ != '_PRON_ 'else word.lower_ for word in mytokens]
    mytokens=[word for word in mytokens if word not in stop_words and word not in punctuations]
    return mytokens

class predictors(TransformerMixin):
    def fit_transform(self, X, **transform_params):
        return [clean_text(text) for text in X]
    def fit(self,X,y=None, **fit_params):
        return  self
    def get_params(self,deep=True):
        return {}
def clean_text(text):
        return text.strip().lower()


# In[28]:


df['text']=df['text'].apply(clean_text)


# In[29]:


#Tfidfvectorizer it is count vectorizer which gives equal weightage to all the words
cv=TfidfVectorizer(max_features=100)
x=cv.fit_transform(df['text'])
df1=pd.DataFrame(x.toarray(),columns=cv.get_feature_names())
df.drop(['text'],axis=1,inplace=True)
main_df=pd.concat([df1,df],axis=1)
main_df.head()


# In[31]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=100,criterion="entropy")
model=rfc.fit(X_train,y_train)


# In[ ]:


#Split dataset into test and train 30% 70% respectively
Y=main_df.iloc[:,-1]
X=main_df.iloc[:,:-1]
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[32]:


# Predicting with a test dataset
print(X_test)


# In[33]:


pred = rfc.predict(X_test)
score=accuracy_score(y_test,pred)
print(score)


# In[34]:


print("Classification Report\n")
print(classification_report(y_test,pred))
print("confusion matrix\n")
print(confusion_matrix(y_test,pred))


# In[ ]:




