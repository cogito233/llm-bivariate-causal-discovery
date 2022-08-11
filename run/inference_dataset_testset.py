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
        if (int(df['label'].iloc[i])!=setup or setup == -1):
            continue
        current_str = df['text'].iloc[i].replace("\\n","\n")
        perfix.append(current_str.split("\n\n\n\n")[0]+"\n\n")
        postfix.append("\n\n"+current_str.split("\n\n\n\n")[-1])
    ic(len(perfix))
    return perfix, postfix

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("setup", help="the selected setup number")
    parser.add_argument("--output", help="the path of output file")
    parser.add_argument("--first", help="use the first 50% of the dataset, otherwise the remain 50%",
                        action="store_true")
    args = parser.parse_args()

    from datasets import load_from_disk
    dataset = load_from_disk('./yelp_review_full_split_train_dev')
    dataset_idx = 0
    if (args.train):
        subdataset = dataset['selected_train']
    elif (args.dev):
        subdataset = dataset['selected_dev']
        dataset_idx += 2
    else:
        raise Exception('You should select a subdataset')
    begin, end = 0, len(subdataset)
    if (args.first):
        end = end//2
    else:
        begin = end//2
        dataset_idx += 1
    ic(begin, end)

    model = ModelWarper()

    print(f"Model.device is: {model.device}")
    setup = int(args.setup)
    ic(setup)
    if (setup<-1 or setup>3):
        raise Exception('incorrect setup number')

    final_dict = {
        "dataset_idx":dataset_idx,
        "setup":setup,
        "label":[],
        "text":[],
        "answer":[],
    }
    import numpy as np

    df = pd.read_csv('yelp_prompt_arguement_full_version.csv')
    perfix_list, postfix_list = generate_prompts(df, setup)
    tokenizer_list = []
    for i in range(len(perfix_list)):
        tokenizer_list.append(TokenizerWarper([perfix_list[i], postfix_list[i]]))
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
    np.save(args.output,a)


