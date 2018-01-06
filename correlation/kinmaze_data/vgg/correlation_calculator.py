import pandas as pd
import numpy as np
import math

#Set the threshold for a positive example
thresh = 0.9999999

#Add the name of the file containing probabilities for each window in each imgage
x = pd.ExcelFile("kinmaze_allprobs.xlsx")


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
	#for each image, get the number of sliding windows which have flowering panicles
	count = getCount(row,thresh)
	li = [row[0],count]
	print li
	ser = pd.Series(li)
	df2 = df2.append(ser,ignore_index=True)
#Set the output file name
writer = pd.ExcelWriter('kinmaze_output.xlsx',engine='xlsxwriter')
df2.to_excel(writer,sheet_name='Sheet1')
writer.save()
