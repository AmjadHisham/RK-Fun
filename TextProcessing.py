import pandas as pd
import string
import re
from string import punctuation
import codecs

def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)

def remove_spaces(x):
    lenx = len(x)
    count = 0
    spaces = ""
    for i in x:
        if i == " ":
            count+=1
            spaces+=" "
        elif i != " " and count > 0:
            x = x.replace(spaces," ")
            count = 0
            spaces = ""
    return x

def strip_punctuation(s):
    s = s.replace("/"," ")
    return ''.join(c for c in s if c not in punctuation)

def remove_rn(s):
    return (",".join([t.rstrip() for t in s.splitlines() ])).replace(r'\t',"").replace(r'\n',"").replace(r'\r',"")


def replace_null(s):
    if pd.isnull(s):
        return ''
    else:
        return s
        
def clean_up(s):
    return remove_spaces(remove_rn(s)).replace(","," ")



def does_exist(text,s):
    if s.lower() in str(text).lower():
        return 1
    else:
        return 0

    
def contains_target_words(text,target_words,match_num):
    
    if match_num==None:
        match_num=len(target_words)
    
    flag=True
    word_count = 0
    for word in target_words:
        has_word = does_exist(text,str(word))
        word_count+=has_word
        
    if word_count<match_num:
        flag=False
        
    return flag

def cosine_similarity(query,target):
    return 0

        
     
    


        
    
    
