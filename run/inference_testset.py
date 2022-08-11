from tqdm import trange
import torch
from icecream import ic
import pandas as pd
from warper import TokenizerWarper, ModelWarper
import argparse

max_length = 1024
def generate_prompts(df, setup):
    perfix = []
    postfix = []
    for i in range(len(df)):
        if (int(df['label'].iloc[i])!=setup):
            continue
        current_str = df['text'].iloc[i].replace("\\n","\n")
        perfix.append(current_str.split("f{review_text}")[0])
        postfix.append(current_str.split("f{review_text}")[-1])
    ic(len(perfix))
    print(perfix)
    print(postfix)
    return perfix, postfix

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("setup", help="the selected setup number")
    parser.add_argument("model", help="the name of selected model")
    parser.add_argument("part", help="use the first, second, third,forth 25%, of the dataset")
    #parser.add_argument("--output", help="the path of output file")
    parser.add_argument("--test", help="use the train dataset", action="store_true")
    parser.add_argument("--augment", help="use the develop dataset", action="store_true")
    args = parser.parse_args()

    from datasets import load_from_disk
    #dataset = load_from_disk('./yelp_review_full_split_train_dev')
    dataset_idx = 0
    if (args.test):
        subdataset = load_from_disk('./origin_dataset_test')
        dataset_name = 'test_finalize'
    elif (args.augment):
        subdataset = load_from_disk('./augmented_dataset_test')
        dataset_name = 'augment_finalize'
        dataset_idx += 2
    else:
        raise Exception('You should select a subdataset')
    begin, end = 0, len(subdataset)
    bounder = [0, end//4, end//2, end*3//4 ,end]
    begin = bounder[int(args.part)-1]
    end = bounder[int(args.part)]
    ic(begin, end)

    model = ModelWarper(model_path = f"./{args.model}_saved/model")
    output_path = f"./intermediate_data/s{args.setup}_{dataset_name}{int(args.part)}_{args.model}.npy"
    ic(output_path)
    import os
    if (os.path.exists(output_path)):
        ic("file already existed")
        exit(0)
    print(f"Model.device is: {model.device}")
    setup = int(args.setup)
    ic(setup)
    if (setup<0 or setup>3):
        raise Exception('incorrect setup number')

    final_dict = {
        "dataset_idx":dataset_idx,
        "setup":setup,
        "label":[],
        "text":[],
        "answer":[],
    }
    import numpy as np

    #df = pd.read_csv('yelp_prompt_example.csv')
    #df = pd.read_csv('finalized_prompt.csv')
    df = pd.read_csv('probing_prompt.csv')
    perfix_list, postfix_list = generate_prompts(df, setup)
    tokenizer_list = []
    for i in range(len(perfix_list)):
        tokenizer_list.append(TokenizerWarper([perfix_list[i], postfix_list[i]], tokenizer_path = f"./{args.model}_saved/token"))
    label_idx = np.array([352, 362, 513, 604, 642])
    sum = 0
    with torch.no_grad():
        for i in trange(begin,end,4):
            batch_data =subdataset[i:min(i+4, end)]
            sum+=1
            if (sum==10):
                #break
                pass
            final_dict['label'].append(batch_data['label'])
            final_dict['text'].append(batch_data['text'])
            inputs = [tokenizer(batch_data['text']) for tokenizer in tokenizer_list]
            outputs = []
            for k in inputs:
                #if (len(k['input_ids'][0])>1024):
                outputs.append(model(**k))

            final_dict["answer"].append(outputs)

    a=np.array(final_dict)
    np.save(output_path,a)



"""
from transformers import AutoTokenizer
from transformers import GPT2LMHeadModel

for model_name in ['gpt2-xl']:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    tokenizer.save_pretrained(f"./{model_name}_saved/token")
    model.save_pretrained(f"./{model_name}_saved/model")
"""