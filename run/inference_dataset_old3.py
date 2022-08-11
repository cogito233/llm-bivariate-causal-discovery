def generate_prompts(df):
    perfix = []
    postfix = []
    for i in range(len(df)):
        current_str = df['text'].iloc[i].replace("\\n","\n")
        perfix.append(current_str.split("\n\n\n\n")[0])
        postfix.append(current_str.split("\n\n\n\n")[-1])
    return perfix, postfix
if __name__=='__main__':
    import pandas as pd
    df = pd.read_csv('yelp_prompt_example.csv')
    """
    perfix_list = ["I just finished eating at a restaurant. Then I opened my Yelp app.\n\nI first gave a rating, and then wrote the following review:\n\n",
                   "I just finished eating at a restaurant. Then I opened my Yelp app, and wrote the following review: \n\n",
                   "I opened my Yelp app, and started to read some reviews of a restaurant that I want to try. I saw a user wrote this review:\n\n"]
    postfix_list = ["\n\nThe review is an explanation of why I gave a rating (out of 1 to 5 stars) of",
                    "\n\nThen I gave the rating. In terms of 1 to 5 stars, I think this restaurant is worth a",
                    "\n\nIn terms of 1 to 5 stars, I think this user rated it a"]
    """
    label_idx = np.array([352, 362, 513, 604, 642])
    with torch.no_grad():
        for i in tqdm(dataset['test']):
            final_dict['label'].append(i['label'])
            final_dict['text'].append(i['text'])
            for j in range(3):
                current_text = perfix_list[j]+i['text']+postfix_list[j]
                input = tokenizer(current_text, return_tensors='pt', padding=True)
                if (len(input['input_ids'][0])>1024):
                    perfix_input = tokenizer(perfix_list[j], return_tensors='pt')
                    text_input = tokenizer(i['text'], return_tensors='pt')
                    postfix_input = tokenizer("..."+postfix_list[j], return_tensors='pt')
                    for k in input:
                        ic([perfix_input[k],text_input[k][:,1024-len(perfix_input[k][0])-len(postfix_input[k][0])],postfix_input[k]])
                        input[k] = torch.cat([perfix_input[k],text_input[k][:,:1024-len(perfix_input[k][0])-len(postfix_input[k][0])],postfix_input[k]],dim=1)
                for k in input:
                    input[k]=input[k].cuda()
                output = model(**input)[0].cpu().detach().numpy()[0][-1][label_idx]
                final_dict[f"setup{j+1}_answer"].append(output)
                #ic(i['label'], output)
    a=np.array(final_dict)
    np.save("/cluster/project/sachan/zhiheng/causal_prompting/intermediate_data/predict_result.npy",a)


    print(generate_prompts(df))
