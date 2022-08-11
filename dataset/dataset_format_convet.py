from datasets import load_from_disk
from icecream import ic
dataset = load_from_disk('./yelp_review_full_split_train_dev')
df_dict = {
    'text':[],
    'label':[]
}
ic(len(dataset['test']))
for i in dataset['test']:
    df_dict['text'].append(i['text'])
    df_dict['label'].append(i['label'])
    if (len(df_dict['text'])==10000):
        pass
import pandas as pd
ic(len(df_dict['text']))
df = pd.DataFrame.from_dict(df_dict)
df.to_csv("yelp_testset.csv")