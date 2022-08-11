from tqdm import trange
import torch
from icecream import ic
import pandas as pd
from warper import TokenizerWarper, ModelWarper
max_length = 1024
def generate_prompts(df):
    perfix = []
    postfix = []
    for i in range(len(df)):
        current_str = df['text'].iloc[i].replace("\\n","\n")
        perfix.append(current_str.split("\n\n\n\n")[0]+"\n\n")
        postfix.append("\n\n"+current_str.split("\n\n\n\n")[-1])
    return perfix, postfix

if __name__=='__main__':
    from datasets import load_from_disk
    dataset = load_from_disk('./yelp_review_full_split_train_dev')

    model = ModelWarper()

    print(f"Model.device is: {model.device}")

    final_dict = {
        "label":[],
        "text":[],
        "answer":[],
    }
    import numpy as np

    df = pd.read_csv('yelp_prompt_arguement_full_version.csv')
    perfix_list, postfix_list = generate_prompts(df)
    tokenizer_list = []
    for i in range(len(perfix_list)):
        tokenizer_list.append(TokenizerWarper([perfix_list[i], postfix_list[i]]))
    label_idx = np.array([352, 362, 513, 604, 642])
    sum = 0
    with torch.no_grad():
        for i in trange(0,len(dataset['selected_train']),8):
            batch_data = dataset['selected_train'][i:i+8]
            sum+=1
            if (sum==10):
                #break
                pass
            final_dict['label'].append(batch_data['label'])
            final_dict['text'].append(batch_data['text'])
            inputs = [tokenizer(batch_data['text']) for tokenizer in tokenizer_list]
            outputs = []
            for k in inputs:
                outputs.append(model(**k))

            final_dict["answer"].append(outputs)

        for i in trange(0,len(dataset['selected_dev']),8):
            batch_data = dataset['selected_dev'][i:i+8]
            sum+=1
            if (sum==10):
                #break
                pass
            final_dict['label'].append(batch_data['label'])
            final_dict['text'].append(batch_data['text'])
            inputs = [tokenizer(i['text']) for tokenizer in tokenizer_list]
            outputs = []
            for k in inputs:
                outputs.append(model(k))

            final_dict["answer"].append(outputs)
    a=np.array(final_dict)
    np.save("/cluster/project/sachan/zhiheng/causal_prompting/intermediate_data/predict_train_14.npy",a)


