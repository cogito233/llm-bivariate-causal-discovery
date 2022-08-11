from tqdm import tqdm
import torch
from icecream import ic
import pandas as pd
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
    #dataset.save_to_disk('./yelp_review_full')

    from transformers import AutoTokenizer
    from transformers import GPT2LMHeadModel
    tokenizer = AutoTokenizer.from_pretrained("./gpt2-large_saved/token")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('./gpt2-large_saved/model')

    #tokenizer.save_pretrained('./gpt2-large_saved/token')
    #model.save_pretrained('./gpt2-large_saved/model')

    device = 'cuda:0'
    model = model.to(device)

    print(f"Model.device is: {model.device}")

    final_dict = {
        "label":[],
        "text":[],
        "answer":[],
    }
    import numpy as np

    df = pd.read_csv('yelp_prompt_arguement_full_version.csv')
    perfix_list, postfix_list = generate_prompts(df)
    """
    perfix_list = ["I just finished eating at a restaurant. Then I opened my Yelp app.\n\nI first gave a rating, and then wrote the following review:\n\n",
                   "I just finished eating at a restaurant. Then I opened my Yelp app, and wrote the following review: \n\n",
                   "I opened my Yelp app, and started to read some reviews of a restaurant that I want to try. I saw a user wrote this review:\n\n"]
    postfix_list = ["\n\nThe review is an explanation of why I gave a rating (out of 1 to 5 stars) of",
                    "\n\nThen I gave the rating. In terms of 1 to 5 stars, I think this restaurant is worth a",
                    "\n\nIn terms of 1 to 5 stars, I think this user rated it a"]
    """
    label_idx = np.array([352, 362, 513, 604, 642])
    perfix_input = []
    for i in perfix_list:
        perfix_input.append(tokenizer(i, return_tensors='pt'))
    postfix_input = []
    for i in postfix_list:
        postfix_input.append(tokenizer("..."+i, return_tensors='pt'))
    sum=0
    ic(dataset['selected_train'][9])
    with torch.no_grad():
        for i in tqdm(dataset['selected_train']):
            sum+=1
            if (sum==10):
                #break
                pass
            final_dict['label'].append(i['label'])
            final_dict['text'].append(i['text'])
            current_text = []
            answer_list = []
            for j in range(len(perfix_list)):
                current_text.append(perfix_list[j]+i['text']+postfix_list[j])
            input = tokenizer(current_text, return_tensors='pt', padding=True)
            #ic(input['input_ids'].shape)
            if (len(input['input_ids'][0])>max_length):
                text_input = tokenizer(i['text'], return_tensors='pt')
                for k in input:
                    input[k] = []
                    for j in range(len(perfix_input)):
                    #perfix_input = tokenizer(perfix_list[j], return_tensors='pt')
                        #postfix_input = tokenizer("..."+postfix_list[j], return_tensors='pt')
                        ic(perfix_input[j][k][0],text_input[k][0,:max_length-len(perfix_input[j][k][0])-len(postfix_input[j][k][0])],postfix_input[j][k][0])
                        input[k].append(torch.cat([perfix_input[j][k][0],text_input[k][0,:max_length-len(perfix_input[j][k][0])-len(postfix_input[j][k][0])],postfix_input[j][k][0]],dim=0))
                    input[k]=torch.cat(input[k]).reshape([-1, max_length]).contiguous()
                    ic(input[k].shape)
            for k in input:
                input[k] = input[k].to(device)
            output = model(**input)[0].cpu().detach().numpy()[:,:, label_idx]
            answer = []
            for i in range(len(perfix_list)):
                selected_output = output[i][input['attention_mask'][i].cpu().detach().numpy()==1][-1]
                answer.append(selected_output)
            #TODO
            #ic(len(answer))
            #ic(answer[0].shape)
            #ic(output.shape)
            final_dict[f"answer"].append(output)
            #ic(i['label'], output)
    a=np.array(final_dict)
    np.save("/cluster/project/sachan/zhiheng/causal_prompting/intermediate_data/predict_train_14.npy",a)


