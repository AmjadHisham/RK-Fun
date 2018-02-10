import pandas as pd
import string
import re
from string import punctuation
from datetime import datetime
import nltk 
import os
import bs4
from time import gmtime, strftime 
import seaborn as sns
import matplotlib.pyplot as plt
import importlib
import TextProcessing
importlib.reload(TextProcessing)



def text_utf8(df):

#     df['Summary']=df['Summary'].apply(lambda x: TextProcessing.replace_null(x))
# 
#     df['Description']=df['Description'].apply(lambda x: TextProcessing.replace_null(x))
    df['Text_UTF8']=df[['Summary','Description']].apply(lambda x: str(x[1])+" "+str(x[0]),axis=1)
#     df['Text_UTF8']=df[['Text_UTF8']].apply(lambda x: TextProcessing.clean_up(x))
    return df


def subset_phone(df,phone_list):
    new = pd.DataFrame()
    for phone in phone_list:
        new=pd.concat([new,df[df.Phone==phone]])
    return new

def subset_text_cs_utf8(df,target_words,match_num=None):
    mask = df['Text_UTF8'].apply(lambda x: TextProcessing.contains_target_words(x,target_words,match_num) )
    return df[mask]

def subset_ad_id(df,ad_list):
    new = pd.DataFrame()
    for ad in ad_list:
        new=pd.concat([new,df[df.Ad_ID==ad]])
    return new


    hour = df.Ad_Time[:].apply(lambda x: DatetimeProcessing.get_posting_hour(x))
    weekday = df.Ad_Date[:].apply(lambda x: DatetimeProcessing.get_weekday(x))

    posting_pattern  =pd.DataFrame()
    posting_pattern['Hour'] = hour
    posting_pattern['Weekday'] = weekday
    
    sns.heatmap(posting_pattern.groupby(['Hour','Weekday']).size().unstack(fill_value=0))
    plt.show()

#def input_note_taking_fields(df):
    #Create column in database for note taking
    #Update the columns
    #UPDATE table_name
    #SET column1 = value1, column2 = value2, ...
    #WHERE condition;
    


    
