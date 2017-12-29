import pandas as pd
import numpy as np
import math
thresh = 0.9999999

x = pd.ExcelFile("kinmaze_all.xlsx")



df = x.parse("Sheet1")

df2 = pd.DataFrame()

def getCount(row,thresh):
	count = 0
	for i in range(1,len(row)):
	  if not math.isnan(row[i]):
	    if row[i] >= thresh:
	       count += 1
	return count
	


for index,row in df.iterrows():
	count = getCount(row,thresh)
	li = [row[0],count]
	print li
	ser = pd.Series(li)
	df2 = df2.append(ser,ignore_index=True)

writer = pd.ExcelWriter('kinmaze_all_almost1.xlsx',engine='xlsxwriter')
df2.to_excel(writer,sheet_name='Sheet1')
writer.save()
