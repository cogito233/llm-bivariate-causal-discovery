import jsonlines
from icecream import ic
import pandas as pd
def load_jsonl_list(path):
    jsonl_list = []
    with jsonlines.open(path, 'r') as reader:
        for i in reader:
            jsonl_list.append(dict(i))
    return jsonl_list

def extract_and_save_dataset(results):
    from datasets import Dataset
    dataset_dict = {
        "text":[],
        "label":[],
        "origin_data_idx":[],
        "origin_text":[],
    }
    idx_based_dict = {}
    for i in range(50000):
        idx_based_dict[i] = []
    for result in results:
        for result_text in result['result']:
            if (len(idx_based_dict[result['idx']])<10 and not (result_text in idx_based_dict[result['idx']])):
                idx_based_dict[result['idx']].append(result_text)
                dataset_dict['text'].append(result_text)
                dataset_dict['label'].append(result['label'])
                dataset_dict['origin_data_idx'].append(result['idx'])
                dataset_dict['origin_text'].append(result['text'])

    ic(len(dataset_dict['text']))
    df = pd.DataFrame.from_dict(dataset_dict)
    df.to_csv("augmented_dataset_test_full.csv")
    from datasets import Dataset
    dataset = Dataset.from_dict(dataset_dict)
    dataset.save_to_disk("augmented_dataset_test_full")
    ic(dataset)

    sum = 0
    li = []
    for i in idx_based_dict:
        if len(idx_based_dict[i])<10:
            sum+=(10-len(idx_based_dict[i])+3)//4
            for j in range((10-len(idx_based_dict[i])+3)//4):
                li.append(i)
    ic(li[0:100])
    ic(sum)
    import numpy as np
    a = np.array(li)
    np.save("regenerate_dataset.npy", a)


if __name__ == "__main__":
    base_path = "/cluster/project/sachan/zhiheng/causal_prompting/intermediate_data/"
    count_dict = {}
    for i in range(50000):
        count_dict[i]=0
    result_list = []

    for postfix in ["", "_1", "_2"]:
        for perfix in range(1,6,1):
            file_name = f"augment_test_{perfix}{postfix}.jsonl"
            x = load_jsonl_list(base_path+file_name)
            for augment_result in x:
                count_dict[augment_result['idx']]+=4
                result_list.append(augment_result)
    import json

    file_name = f"regenerating_dataset.jsonl"
    f = open(base_path + file_name)
    lines = f.readlines()
    import json
    for augment_json in lines:
        try:
            augment_result = json.loads(augment_json)
        except:
            continue
        count_dict[augment_result['idx']] += 4
        result_list.append(augment_result)
    """
    sum = 0
    li = []
    for i in count_dict:
        if count_dict[i]<10:
            sum+=(10-count_dict[i]+3)//4
            for j in range((10-count_dict[i]+3)//4):
                li.append(i)
    #ic(li[0:100])
    ic(sum)
    import numpy as np
    a = np.array(li)
    np.save("regenerate_dataset.npy", a)
    """
    extract_and_save_dataset(result_list)
