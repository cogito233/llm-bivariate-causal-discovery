from datasets import load_from_disk
from icecream import ic

dataset_qwq = load_from_disk('./yelp_review_full_split_train_dev')
dataset = []
for i in dataset_qwq['test']:
    dataset.append((i['text'], i['label']))
ic(dataset[0:3])
import textattack
dataset = textattack.datasets.Dataset(dataset)
ic(dataset)