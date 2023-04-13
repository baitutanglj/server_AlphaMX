import pandas as pd
import shutil


train_id = pd.read_csv('../data/train_id.csv')
for id in train_id['0']:
   shutil.copy('../data/whole_data/'+id+'.acfc_data','../data/train_data')
   
test_id = pd.read_csv('../data/test_id.csv')
for id in test_id['0']:
   shutil.copy('../data/whole_data/'+id+'.acfc_data','../data/test_data')
   
   
valid_id = pd.read_csv('../data/valid_id.csv')
for id in valid_id['0']:
   shutil.copy('../data/whole_data/'+id+'.acfc_data','../data/valid_data')