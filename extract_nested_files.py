import os 
from os import walk
import sys

parent_dir = "train"

print("parent directory: ", parent_dir)

# Contains full path to files
file_list = []

class Parent():

    def __init__(self, parent_dir):
        self.dir_ = parent_dir
    
    def make_children(self):
        sub_dir = []
        for _,d,f in walk(self.dir_):
            sub_dir.extend(d)
            break
            
        return sub_dir

    def get_files(self):
        files = []
        for _,d,f in walk(self.dir_):
            files.extend(f)
            break
        return [self.dir_+'/'+f for f in files]
    


# All parents are children

dir_list =[parent_dir]

while True:
    
    children =[]
    
    for parent in dir_list:
        
        child = Parent(parent)
        file_list+=child.get_files()
        children.append(child)
        
    
    dir_list =[]
    
    for child in children:
        parent_dir = child.dir_
        
        for c in child.make_children():
            dir_list.append(parent_dir+"/"+c)
            

    if len(dir_list) < 1:
        break
        
        
# Logic to create dictionary out of file list
import importlib
import MapReduce
import sys
import re


mr = MapReduce.MapReduce()


def mapper(record):

    x = re.search('\((.+?)\)',record)
    mr.emit_intermediate(x.group(0), [record])

def reducer(key, row):
    length = len(row)
    print(length)
    v = []
    for i in range(1,length):
        v += row[i]
    mr.emit({key:v})
    

if __name__ == '__main__':
    mr.execute(file_list, mapper, reducer)
    
    
# Logic to merge dictionaries together
merge_dict = {}
for d in mr.result:
    for k, v in d.items():  # d.items() in Python 3+
        merge_dict.setdefault(k.replace('(','').replace(')',''), []).append(v)
        

        
# Create lut table
lut = {}
i = 0
for key in phone_dict.keys():
    lut[key] = i
    i+=1

phone_dict.keys(), lut

# Create inverse lut table

def invert_dict(d):
    return dict([ (v, k) for k, v in d.items( ) ])

inv_lut = invert_dict(lut)
        
    
    