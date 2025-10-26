#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import nltk
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve,roc_auc_score,classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Dropout,SpatialDropout1D
from tensorflow.keras.layers import Embedding

import streamlit as st

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv("mail_data.csv")
df.head()


# In[4]:


df.duplicated().sum()


# In[3]:


df.drop_duplicates(inplace=True)


# In[7]:


len(df)


# In[6]:


len(df[df['Category']=='ham'])


# In[7]:


len(df[df['Category']=='spam'])


# From the above it is clear that number of spam and ham(not spam) emails are unbalanced.  The number of ham emails are far higher in number which denotes that it is the majority class and number of spam emails are very few compared to ham mails and thus it is the minority class.

# In[4]:


def balance_dataset():
    spam_email=df[df['Category']=='spam']
    ham_email=df[df['Category']=='ham']

    bal_spam_msg=ham_email.sample(n=len(spam_email),random_state=42)
    df_bal=pd.concat([bal_spam_msg,spam_email]).reset_index(drop=True)
    return df_bal


# In[5]:


df_new=balance_dataset()


# In[17]:


sns.countplot(data=df_new,x='Category')
plt.title('Balanced email data')
plt.xticks(ticks=[0,1],labels=['Ham(Not Spam)','Spam'])
plt.show()


# In[11]:


df_new.duplicated().sum()


# In[6]:


punc=string.punctuation
def rem_punc(text):
    temp_text=str.maketrans('','',punc)
    return text.translate(temp_text)
df_new['Message']=df_new['Message'].apply(lambda x:rem_punc(x))
df_new.head()


# In[7]:


def punct_rem():
    for i in range(len(df_new)):
     
        df_new['Message'][i]=df_new['Message'][i].replace('-','')
        df_new['Message'][i]=df_new['Message'][i].replace("£",'')
        df_new['Message'][i]=df_new['Message'][i].replace("...",'')
        
punct_rem()       


# In[21]:


len(df_new)


# In[8]:


def rem_stopwrds(text):
    stopwrds=stopwords.words('english')
    wrds=[]
    for word in str(text).split():
        word=word.lower()
        if word not in stopwrds:
            wrds.append(word)
    mail=" ".join(wrds)
    return mail

df_new['Message']=df_new['Message'].apply(lambda x:rem_stopwrds(x))
df_new.head()


# In[101]:




# ## Tokenization and Padding

# In[9]:


X_train,X_test,Y_train,Y_test=train_test_split(df_new['Message'],df_new['Category'],test_size=0.2,random_state=42)


tokenizer=Tokenizer()
tokenizer.fit_on_texts(X_train)

train_seq=tokenizer.texts_to_sequences(X_train)
test_seq=tokenizer.texts_to_sequences(X_test)

max_len=100

train_seq=pad_sequences(train_seq,maxlen=max_len,padding='post',truncating='post')
test_seq=pad_sequences(test_seq,maxlen=max_len,padding='post',truncating='post')

dict_ham={'ham':0,'spam':1}
Y_train=Y_train.map(dict_ham)
Y_test=Y_test.map(dict_ham)


# ## Modeling and Evaluation

# In[10]:


model=tf.keras.models.Sequential([tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1,output_dim=32,input_length=max_len),
                          tf.keras.layers.LSTM(16),
                          tf.keras.layers.Dense(32,activation='relu'),
                          tf.keras.layers.Dense(1,activation='sigmoid')
                                 ])
model.build(input_shape=(None,100))
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()


# In[11]:


EarlStop=EarlyStopping(patience=3,monitor='val_accuracy',restore_best_weights=True)
lr=ReduceLROnPlateau(patience=2,monitor='val_loss',factor=0.5,verbose=0)

history=model.fit(train_seq,Y_train,validation_data=(test_seq,Y_test),epochs=20,batch_size=32,callbacks=[lr,EarlStop]
                 )


# In[12]:


test_loss,test_accuracy=model.evaluate(test_seq,Y_test)
print('Test Loss : ',test_loss)
print('Test Accuracy : ',test_accuracy)


# Training accuracy is 48%.

# In[112]:


        


# ### Logistic Regression

# In[13]:


feature_extraction=TfidfVectorizer(min_df=1,lowercase=True)


# In[14]:


X_train_features=feature_extraction.fit_transform(X_train)
X_test_features=feature_extraction.transform(X_test)


# In[15]:


lr=LogisticRegression()
lr.fit(X_train_features,Y_train)


# In[16]:


y_pred_lrtrain=lr.predict(X_train_features)
print('Accuracy score on training data is :',accuracy_score(Y_train,y_pred_lrtrain))


# In[17]:


y_pred_lrtest=lr.predict(X_test_features)
print('Accuracy score on test data is :',accuracy_score(Y_test,y_pred_lrtest))


# In[21]:


# Test the model with custom email messages
def spam_detect(email):
    
    email_features=feature_extraction.transform([email])
    pred=lr.predict(email_features)
    
    if (pred)[0]==0:
        return 'Ham mail'
        
    else:
        return 'Spam mail'
        


# In[23]:




# In[83]:


cm=confusion_matrix(Y_test,y_pred_lrtest)

plt.figure(figsize=(6,4))
sns.heatmap(cm,annot=True,fmt="d",cmap='Blues',cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# 144 samples out of 167 are correctly classified.  Recall is high to about 96% for class 0 and 85% for class 1.  

# In[85]:


print('Classification Report \n',classification_report(Y_test,y_pred_lrtest))


# ## Web deployment

# In[ ]:


st.title('Email Classification System')
with st.form('Email classifier'):
	col1,col2=st.columns([2,1])
with col1:
	email_input=st.text_input('Enter email contents here: ',placeholder='Enter email content')
with col2:
	submit=st.form_submit_button("Check spam or not")
if submit:
  mail_type=spam_detect(email_input) 	
	
  st.write(mail_type)  


