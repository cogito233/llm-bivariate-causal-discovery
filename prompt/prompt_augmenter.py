from textattack.augmentation.recipes import EasyDataAugmenter
from tqdm import trange
import multiprocessing
import argparse
from icecream import ic
from multiprocessing import Process, Queue
import jsonlines
import time
import random
import numpy as np
import nltk
import nltk.translate.gleu_score as gleu
import numpy
import os

try:
  nltk.data.find('tokenizers/punkt')
except LookupError:
  nltk.download('punkt')

class MyProcess(Process): #继承Process类
    def __init__(self, name, sub_dataset, path):
        super(MyProcess,self).__init__()
        self.name = name
        self.augmenter = EasyDataAugmenter()
        self.augmenter.fast_augment = True
        self.augmenter.high_yield = True
        self.augmenter.transformations_per_example = 20
        self.sub_dataset = sub_dataset
        self.Q = q
        self.path = path

    def run(self):
        for datapoint in self.sub_dataset:
            while True:
                try:
                    result = self.augmenter.augment(datapoint['text'])
                    break
                except:
                    print(f"A error occured at {self.name}")
                    time.sleep(random.random())
            datapoint['result'] = result
            datapoint['name'] = self.name
            ic(self.name)
            while True:
                try:
                    with jsonlines.open(self.path, 'a') as writer:
                        writer.write(datapoint)
                    break
                except:
                    time.sleep(random.random())
            ic(self.name+" finish write file")



def augment_single_sentence(str, max_transformation_words, number_exapmles):
    #ic(str)
    augmenter = EasyDataAugmenter(pct_words_to_swap=max_transformation_words, transformations_per_example = number_exapmles)
    augmenter.fast_augment = True
    augmenter.high_yield = True
    #augmenter.transformations_per_example = max_transformation_words
    result = augmenter.augment(str)
    return result

# use MT metric to evaluate the distance bewteen the augmented sentence and origin sentence
def evaluate_distance(hyp, ref_b):
    hyp = hyp.split()
    ref_b = ref_b.split()
    score_1to4grams = gleu.sentence_gleu([ref_b], hyp, min_len=1, max_len=4)
    return score_1to4grams


def generate_augmented_sentence(perfix, postfix):
    # input is a prompt, and output is [4, 10, 2] list, (score, GLUE score)
    augmented_list = []
    prompt = perfix+' f{review_text} '+postfix
    for max_transformation_words in [0.05, 0.1, 0.2, 0.4]:
        perfix_augmented_example = augment_single_sentence(perfix, max_transformation_words, 16)
        postfix_augmented_example = augment_single_sentence(postfix, max_transformation_words, 16)
        for i in range(min(len(perfix_augmented_example),len(postfix_augmented_example) )):
            prompt_aug = perfix_augmented_example[i]+' f{review_text} '+postfix_augmented_example[i]
            augmented_list.append((evaluate_distance(prompt, prompt_aug), prompt_aug))
    augmented_list = sorted(augmented_list, reverse=True)
    ic(len(augmented_list))
    #ic(augmented_list[0])
    #ic(augmented_list[1])
    #ic(augmented_list[-2])
    #ic(augmented_list[-1])
    result_list = []
    for i in range(0,len(augmented_list),16):
        result_list.append(augmented_list[i:i+10])
    return result_list
# 1. split the prompt by perfix and postfix
# 2. augment them seperately
# 3. concate them

if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('smaller_prompt_pool.csv')
    ic(len(df))
    from warper import generate_prompts
    perfix_list, postfix_list = generate_prompts(df, 0)
    df_dict = {
        "text":[],
        "label":[],
        "score":[],
        "group":[],
    }
    for i in range(6):
        augmented_dataset = generate_augmented_sentence(perfix_list[0], postfix_list[0])
        for group_number in range(4):
            for k in range(10):
                df_dict['group'].append(group_number+1)
                df_dict['text'].append(augmented_dataset[group_number][k][1])
                df_dict['score'].append(augmented_dataset[group_number][k][0])
                df_dict['label'].append(df['label'].iloc[i])

    new_df = pd.DataFrame.from_dict(df_dict)
    new_df.to_csv('prompt_pool_augmented.csv')


