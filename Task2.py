#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Summary of all properties from all tabs in excel file
import pandas as pd
import numpy as np
from xlrd import open_workbook
from openpyxl.reader.excel import load_workbook

file_name = "C:\\Users\\vyshnavi.garimella\\Desktop\\G_Tubes_Result_v0.xlsx"
workbook = open_workbook(file_name)

df = pd.DataFrame(columns=['Frame','Property', 'Count', 'Anomaly', '#technologies','Technology1', 'Technology2', 'Technology3', 'Technology4', 'Technology5', 'Technology6'])
simple_list=[]
k=0

for sheet in workbook.sheets():
    if(sheet.name == ("Layer Material") or sheet.name == ("Component Properties") or sheet.name == ("Package Details") or sheet.name == ("Wall Thickness")):
        dfs = pd.read_excel(file_name, sheet.name)
        
        if(sheet.name == ("Wall Thickness")):
            dfs = dfs.rename(columns={'*Wall Thickness_Property':'Property', "*Wall Thickness_Technology":"Technology"})

        property_set = set(dfs['Property'])

        simple_list=[]

        for val in (property_set):
        
            j = 0
            i = 0

            count = dfs.loc[dfs.Property == val, 'Property'].count()

            dfs1 = dfs[(dfs['Property'] == val)]

            anomalyCount = dfs1.loc[dfs1.Anomaly == 'Yes', 'Anomaly'].count()

            technology_set = set(dfs1['Technology'])

            technologyCount = len(technology_set)

            simple_list.append(list(technology_set))

            df = df.append({'Frame': sheet.name, 'Property': val, 'Count': count, 'Anomaly': anomalyCount, '#technologies': technologyCount }, ignore_index=True)
        
        for i in range(k, len(property_set)+k):
            
            j = 0

            while j < len(simple_list[i-k]): 

                str1="Technology"+str(j+1)
               
                df[str1][i]=simple_list[i-k][j]

                j = j+1        
                
        k=k+len(property_set)

