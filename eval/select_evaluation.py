# This code is writed for train all the offset and testing the accuracy of all the prompts
# and select the best one
# 1. Concate the structure from the
#TODO
import numpy as np
import sklearn.metrics
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
from icecream import ic
import pandas as pd
from tqdm import trange
final_answer = []

def generate_prompts(df):
    prompt_list = []
    for i in range(len(df)):
        current_str = df['text'].iloc[i].replace("\\n","\n")
        perfix = current_str.split("f{review_text}")[0]
        postfix = current_str.split("f{review_text}")[-1]
        label =  df['group'].iloc[i]
        prompt_list.append({
            'perfix':perfix,
            'postfix':postfix,
            'label':label,
        })
    return prompt_list

def eval_given_offset(result_list, offset):
    pass

def eval_and_offset(result_list):
    # len(result_list) = 20,000, the first 10,000 is train set, used to generate the offset
    # the last 10,000 is the dev set, used to evaluate the performance
    #ic(result_list[0:10])
    train_list, dev_list = result_list[:len(result_list)//2],result_list[len(result_list)//2:]
    gt_labels = [i[1] for i in train_list]
    predict_prob = np.array([i[0] for i in train_list])
    predict_labels = predict_prob.argmax(axis=1)


    gt_labels_dev = [i[1] for i in dev_list]
    predict_prob_dev = np.array([i[0] for i in dev_list])
    predict_labels_dev = predict_prob_dev.argmax(axis=1)

    print("###########before normalize#############")
    ic(Counter(gt_labels), Counter(predict_labels))
    ic(accuracy_score(gt_labels, predict_labels), f1_score(gt_labels, predict_labels, average='weighted'))

    ic(Counter(gt_labels_dev), Counter(predict_labels_dev))
    ic(accuracy_score(gt_labels_dev, predict_labels_dev), f1_score(gt_labels_dev, predict_labels_dev, average='weighted'))

    offset = np.array([0.0,0.0,0.0,0.0,0.0])
    lr = 0.01
    cgt = dict(Counter(gt_labels))
    for i in range(1000):
        predict_labels = predict_prob.argmax(axis=1)
        predict_labels_dev = predict_prob_dev.argmax(axis=1)
        flag = False
        for label in range(5):
            delta = lr*(1-list(predict_labels).count(label)/cgt[label])
            if (delta!=0):
                flag = True
            offset[label]+=delta
            predict_prob[:,label]+=delta
            predict_prob_dev[:,label]+=delta
            #ic(offset[label], delta, flag)
        if (not flag):
            break

    predict_labels = predict_prob.argmax(axis=1)
    predict_labels_dev = predict_prob_dev.argmax(axis=1)
    print("###########after normalize#############")
    ic(Counter(gt_labels), Counter(predict_labels))
    ic(accuracy_score(gt_labels, predict_labels), f1_score(gt_labels, predict_labels, average='weighted'))
    ic(Counter(gt_labels_dev), Counter(predict_labels_dev))
    ic(accuracy_score(gt_labels_dev, predict_labels_dev),
       f1_score(gt_labels_dev, predict_labels_dev, average='weighted'))
    return offset, accuracy_score(gt_labels, predict_labels), accuracy_score(gt_labels_dev, predict_labels_dev)
def renormalize_data(prompt_list, origin_data_paths = "/cluster/project/sachan/zhiheng/causal_prompting/intermediate_data/"):
    #dim1, setup number, 3; dim2, number of prompt, 10; dim3, 20000, prediction of each prompt
    whole_dataset = [[],[],[],[]]
    for setup_number in [1,2,3]:
        for prompt in prompt_list:
            if (prompt['label'] == setup_number):
                whole_dataset[setup_number].append([])
        for subdataset_name in ['train', 'dev']:
            for sub_dataset in [1,2]:
                file_name = f"s{setup_number}_{subdataset_name}{sub_dataset}_selected.npy"
                result_path = origin_data_paths + file_name
                ic(result_path)
                a = np.load(result_path, allow_pickle=True)
                result_dict = a.tolist()
                ic(result_dict.keys())
                for batch_id in trange(len(result_dict['answer'])):
                    for prompt_id in range(len(result_dict['answer'][batch_id])):
                        batch_label = result_dict['label'][batch_id]
                        batch_predict = result_dict['answer'][batch_id][prompt_id]
                        #ic(batch_label, batch_predict)
                        for i in range(len(batch_label)):
                            whole_dataset[setup_number][prompt_id].append((batch_predict[i], batch_label[i]))

        for prompt_id in range(len(whole_dataset[setup_number])):
            #ic(whole_dataset[setup_number][prompt_id][0:10])
            offset, train_accuracy, accuracy = eval_and_offset(whole_dataset[setup_number][prompt_id])
            print(f"For setup {setup_number}, prompt {prompt_id}, the final accuracy is {accuracy}, offset is {offset}")
            final_answer.append(f"For setup {setup_number}, prompt {prompt_id},the train accuracy is {train_accuracy}, the dev accuracy is {accuracy}")

if __name__=='__main__':
    df = pd.read_csv('selected_prompt.csv')
    prompt_list = generate_prompts(df)
    renormalized_data = renormalize_data(prompt_list)
    sum = 0
    for i in final_answer:
        sum+=1
        print(sum, i)