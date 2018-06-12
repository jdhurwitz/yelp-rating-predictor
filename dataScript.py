import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

path = 'yelp-recsys-2013/yelp_training_set/yelp_training_set_'
savepath = 'yelp-recsys-2013/yelp_training_set/'



def write_to_file(sp, filename, df):
	print("creating file: ", filename)
	rows, cols = df.shape
	with open(sp+filename, 'w') as f:
		i = 0
		for index, row in df.iterrows():
			i+=1
			label = str(int(row['stars']) - 1) #shift down
			review = row['text']
			if label=='' or review == '':
				print(label, review)
				continue

			split_review = review.replace('\n', '')
			split_review = split_review.replace('\r', '')
			if split_review[-1:] == '\n':
				split_review = split_review[:-1]
#			merged_review = ' '.join(split_review)
			str_to_write = label+'\t'+split_review
			if i == rows:
				f.write(str_to_write)
			else:
				f.write(str_to_write+'\n')



df_reviews = pd.read_json(path+'review.json', lines=True)

#filter out empty reviews
mask = df_reviews['text'].str.len() > 0
df_reviews = df_reviews.loc[mask]

#df_business = pd.read_json(path+'business.json', lines=True)

#find unique businesses
"""
unique_businesses = df_reviews['business_id'].unique()
train_bus, test_bus = train_test_split(unique_businesses, test_size=0.15)
train_bus, val_bus = train_test_split(train_bus, test_size=0.15)

print(train_bus.shape)
print(test_bus.shape)
print(val_bus.shape)

df_reviews_train = df_reviews.loc[df_reviews['business_id'].isin(train_bus)]
df_reviews_test = df_reviews.loc[df_reviews['business_id'].isin(test_bus)]
df_reviews_val = df_reviews.loc[df_reviews['business_id'].isin(val_bus)]
"""

train, test = train_test_split(df_reviews, test_size = 0.15)
train, val = train_test_split(train, test_size = 0.15)

print(train.shape)
print(test.shape)
print(val.shape)


write_to_file(savepath, 'train.txt', train)
write_to_file(savepath, 'test.txt', test)
write_to_file(savepath, 'dev.txt', val)