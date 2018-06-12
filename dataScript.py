import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

path = 'yelp-recsys-2013/yelp_training_set/yelp_training_set_'
savepath = 'yelp-recsys-2013/yelp_training_set/'



def write_to_file(sp, filename, df):
	print("creating file: ", filename)
	with open(sp+filename, 'w') as f:
		for index, row in df.iterrows():
			label = str(int(row['stars']))
			review = row['text']
			split_review = [token for token in review.split('\n')]
			merged_review = ' '.join(split_review)
			str_to_write = label+'\t'+merged_review
			f.write(str_to_write+'\n')


df_reviews = pd.read_json(path+'review.json', lines=True)

#filter out empty reviews
mask = df_reviews['text'].str.len() > 0
df_reviews = df_reviews.loc[mask]

df_business = pd.read_json(path+'business.json', lines=True)

#find unique businesses
unique_businesses = df_reviews['business_id'].unique()
train_bus, test_bus = train_test_split(unique_businesses, test_size=0.15)
train_bus, val_bus = train_test_split(train_bus, test_size=0.15)

print(train_bus.shape)
print(test_bus.shape)
print(val_bus.shape)

df_reviews_train = df_reviews.loc[df_reviews['business_id'].isin(train_bus)]
df_reviews_test = df_reviews.loc[df_reviews['business_id'].isin(test_bus)]
df_reviews_val = df_reviews.loc[df_reviews['business_id'].isin(val_bus)]

write_to_file(savepath, 'test.txt', df_reviews_test)
write_to_file(savepath, 'train.txt', df_reviews_train)
write_to_file(savepath, 'dev.txt', df_reviews_val)