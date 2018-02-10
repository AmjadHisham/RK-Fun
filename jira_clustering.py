import codecs
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from string import punctuation
import subprocess

import importlib
import Target
importlib.reload(Target)

import webbrowser

# ie = webbrowser.get(webbrowser.Chrome('chrome'))


sq = pd.read_csv("support_que.csv",encoding='latin-1')

sq = Target.text_utf8(sq)
df = Target.subset_text_cs_utf8(sq,['pixel','clock'],match_num=2)
# file = open("standard_quote",'w')
flag = False
for i in df.index[:5]:
#     file.write(str(df.ix[i].Link) + "\n")
    print(str(df.ix[i].Link))
  
should_open = input("Should Open Links? ")
if should_open == "True":
    for i in df.index[:5]: 
        if flag==False:
            webbrowser.open(str(df.ix[i].Link))
            flag=True
        else:
            webbrowser.open_new_tab(str(df.ix[i].Link))

# file.close()

