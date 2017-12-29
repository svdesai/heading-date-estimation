import pandas as pd
import numpy as np
import math
import sys
thresh = float(sys.argv[1])
otpt = sys.argv[2]

print thresh,type(thresh)
print otpt,type(otpt)

x = pd.ExcelFile("kinmaze_siftsvm_newds_1.xlsx")



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

writer = pd.ExcelWriter(otpt,engine='xlsxwriter')
df2.to_excel(writer,sheet_name='Sheet1')
writer.save()
