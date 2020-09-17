#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from openpyxl import load_workbook
file_name = r'C:\Users\Palak\Desktop\G-Tubes_1.xlsx'
sheet_name = 'Component Properties'
dfs = pd.read_excel(file_name, sheet_name)
df = pd.DataFrame(columns=['Template Object','Technology', 'Property', 'Range', 'Uom', 'Remarks'])
workbook = load_workbook(file_name)
worksheet = workbook[sheet_name]

for i in range(len(dfs['Mandatory'])):
    if (dfs['Mandatory'][i] == 1):
        if ((dfs['Property'][i] == '*Bore Diameter') & (dfs['UOM'][i] == 'mm') & (dfs['Target'][i] >= 5) & (dfs['Target'][i] <= 20)):
            df = df.append({'Template Object': dfs['Template Object'][i], 'Technology': dfs['Technology'][i], 'Property': dfs['Property'][i], 'Range': dfs['Target'][i], 'Uom': dfs['UOM'][i], 'Remarks': 'Pass'}, ignore_index=True)
        elif ((dfs['Property'][i] == '*Burst Strength') & (dfs['UOM'][i] == 'bar') & (dfs['Target'][i] > 2)):
             df = df.append({'Template Object': dfs['Template Object'][i], 'Technology': dfs['Technology'][i], 'Property': dfs['Property'][i], 'Range': dfs['Target'][i], 'Uom': dfs['UOM'][i], 'Remarks': 'Pass'}, ignore_index=True)
    
        elif ((dfs['Property'][i] == '*Lamination Strength') & (dfs['Technology'][i] == 'Tube - Laminate Snap-On') & (dfs['UOM'][i] == 'g/15mm') & (dfs['Target'][i] >= 500) & (dfs['Target'][i] <= 1500)):
             df = df.append({'Template Object': dfs['Template Object'][i], 'Technology': dfs['Technology'][i], 'Property': dfs['Property'][i], 'Range': dfs['Target'][i], 'Uom': dfs['UOM'][i], 'Remarks': 'Pass'}, ignore_index=True)
    
        elif ((dfs['Property'][i] == '*Lamination Strength') & (dfs['Technology'][i] == 'Tube - Laminate Threaded') & (dfs['UOM'][i] == 'g/15mm') & (dfs['Target'][i] >= 400) & (dfs['Target'][i] <= 600)):
             df = df.append({'Template Object': dfs['Template Object'][i], 'Technology': dfs['Technology'][i], 'Property': dfs['Property'][i], 'Range': dfs['Target'][i], 'Uom': dfs['UOM'][i], 'Remarks': 'Pass'}, ignore_index=True)
      
        elif ((dfs['Property'][i] == '*Overlap Width') & (dfs['UOM'][i] == 'mm') & (dfs['Target'][i] >= 1.5) & (dfs['Target'][i] <= 2.5)):
             df = df.append({'Template Object': dfs['Template Object'][i], 'Technology': dfs['Technology'][i], 'Property': dfs['Property'][i], 'Range': dfs['Target'][i], 'Uom': dfs['UOM'][i], 'Remarks': 'Pass'}, ignore_index=True)
        
        elif ((dfs['Property'][i] == '*Removal Force') & (dfs['UOM'][i] == 'N') & (dfs['Target'][i] >= 0.1) & (dfs['Target'][i] <= 15)):
            df = df.append({'Template Object': dfs['Template Object'][i], 'Technology': dfs['Technology'][i], 'Property': dfs['Property'][i], 'Range': dfs['Target'][i], 'Uom': dfs['UOM'][i], 'Remarks': 'Pass'}, ignore_index=True)
    
        elif ((dfs['Property'][i] == '*Removal Torque') & (dfs['UOM'][i] == 'N*m') & (dfs['Target'][i] >= 0.01) & (dfs['Target'][i] <= 15)):
            df = df.append({'Template Object': dfs['Template Object'][i], 'Technology': dfs['Technology'][i], 'Property': dfs['Property'][i], 'Range': dfs['Target'][i], 'Uom': dfs['UOM'][i], 'Remarks': 'Pass'}, ignore_index=True)
        
        elif ((dfs['Property'][i] == '*Side Seam Compression') & (dfs['UOM'][i] == '%') & (dfs['Target'][i] >= 15) & (dfs['Target'][i] <= 35)):
            df = df.append({'Template Object': dfs['Template Object'][i], 'Technology': dfs['Technology'][i], 'Property': dfs['Property'][i], 'Range': dfs['Target'][i], 'Uom': dfs['UOM'][i], 'Remarks': 'Pass'}, ignore_index=True)
        
        elif ((dfs['Property'][i] == '*Wall Thickness') & (dfs['UOM'][i] == 'mm') ):   
            df = df.append({'Template Object': dfs['Template Object'][i], 'Technology': dfs['Technology'][i], 'Property': dfs['Property'][i], 'Range': dfs['Target'][i], 'Uom': dfs['UOM'][i], 'Remarks': 9}, ignore_index=True)
        
        else:
            df = df.append({'Template Object': dfs['Template Object'][i], 'Technology': dfs['Technology'][i], 'Property': dfs['Property'][i], 'Range': dfs['Target'][i], 'Uom': dfs['UOM'][i], 'Remarks': 'Anomaly'}, ignore_index=True)
       
    
df.to_excel(r'C:\Users\Palak\Desktop\newdata.xlsx')



file_name = r'C:\Users\Palak\Desktop\G-Tubes_1.xlsx'
sheet_name = 'Materials'
df = pd.read_excel(file_name, sheet_name)
s = df[(df['Property'] == ("Layer-Material"))]
s1 = df[(df['Property'] == ("*Layer-Material"))]
s= s.append(s1, ignore_index = True)
print(s)

s.to_excel(r'C:\Users\Palak\Desktop\newdata1.xlsx')

s5 = s[(s['UOM'] == "µm")]
s5.to_excel(r'C:\Users\Palak\Desktop\newdata2.xlsx')

s7 = s5.groupby('Template Object', as_index=False)['UOM'].count()
s7.to_excel(r'C:\Users\Palak\Desktop\newdata5.xlsx')

test_set = set(s5['Template Object'])
# Creating a set using string 
test_set = set(s5['Template Object'])
i = 0
# Iterating using for loop 
for val in test_set: 
    print(val) 
    i = i+1
print(i)

s6 = s5.groupby('Template Object', as_index=False)['Target'].sum()
            
s6['Target'] = s6['Target']/1000
s6.to_excel(r'C:\Users\Palak\Desktop\newdata3.xlsx')

df = pd.read_excel(r'C:\Users\Palak\Desktop\newdata.xlsx')
for i in range(len(df['Template Object'])):
    if ((df['Property'][i] == '*Wall Thickness')):   
        for j in range(len(s6['Template Object'])):
            print("hi")
            if(df['Template Object'][i] == s6['Template Object'][j]):
                print("hii")
                df['Range'][i] = s6['Target'][j]
df.to_excel(r'C:\Users\Palak\Desktop\newdata4.xlsx') 

