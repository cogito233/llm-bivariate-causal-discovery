from icecream import ic
import pandas as pd
from tqdm import trange
import numpy as np
from collections import Counter
from icecream import ic
import pandas as pd
import sklearn.metrics
from sklearn.metrics import accuracy_score, f1_score

def align_dataset(model_name, subdataset_name, base_dataset, prompt_list, out_dataset_path):
    # dim1, setup number, 3; dim2, number of prompt, n; dim3, 10000, prediction of each prompt
    dataset_dict = base_dataset[:]

    origin_data_paths = "./intermediate_data/"

    whole_dataset = [[],[],[],[]]
    for setup_number in [1,2,3]:
        for prompt in prompt_list:
            if (prompt['label'] == setup_number):
                whole_dataset[setup_number].append([])
        for sub_dataset in ["", "1", "2", "3", "4"]:
            file_name = f"s{setup_number}_{subdataset_name}{sub_dataset}_{model_name}.npy"
            result_path = origin_data_paths + file_name
            #ic(result_path)
            import numpy as np
            try:
                a = np.load(result_path, allow_pickle=True)
            except:
                ic("there is no " + result_path)
                continue

            result_dict = a.tolist()
            #ic(result_dict.keys())
            for batch_id in trange(len(result_dict['answer'])):
                for prompt_id in range(len(result_dict['answer'][batch_id])):
                    batch_label = result_dict['label'][batch_id]
                    batch_predict = result_dict['answer'][batch_id][prompt_id]
                    # ic(batch_label, batch_predict)
                    for i in range(len(batch_label)):
                        whole_dataset[setup_number][prompt_id].append(list(batch_predict[i]))
    dataset_dict['result'] = []
    for i in range(len(dataset_dict['label'])):
        dataset_dict['result'].append([])
        for setup_number in range(1, 4, 1):
            for prompt_id in range(2):
                #ic(setup_number, prompt_id)
                #ic(whole_dataset[setup_number][prompt_id])
                if (len(whole_dataset[setup_number][prompt_id])>i):
                    dataset_dict['result'][-1].append(whole_dataset[setup_number][prompt_id][i])
                else:
                    dataset_dict['result'][-1].append(None)
    from datasets import Dataset
    dataset = Dataset.from_dict(dataset_dict)
    ic(dataset[-3:])
    ic(dataset)
    dataset.save_to_disk(out_dataset_path)
    return dataset

def load_whole_result(model_name, subdataset_name):
    # dim1, setup number, 3; dim2, number of prompt, n; dim3, 10000, prediction of each prompt
    origin_data_paths = "./intermediate_data/"
    whole_dataset = [[],[],[],[],[]]
    for setup_number in [1,2,3,4]:
        for prompt in prompt_list:
            if (prompt['label'] == setup_number):# Origin version is group
                whole_dataset[setup_number].append([])
        for sub_dataset in ["", "1", "2", "3", "4"]:
            file_name = f"s{setup_number}_{subdataset_name}{sub_dataset}_{model_name}_pools_aug.npy"
            result_path = origin_data_paths + file_name
            #ic(result_path)
            import numpy as np
            try:
                a = np.load(result_path, allow_pickle=True)
            except:
                ic("there is no " + result_path)
                continue

            result_dict = a.tolist()
            #ic(result_dict.keys())

            ic(len(result_dict['answer'][0]))
            for batch_id in trange(len(result_dict['answer'])):
                for prompt_id in range(len(result_dict['answer'][batch_id])):
                    batch_label = result_dict['label'][batch_id]
                    batch_predict = result_dict['answer'][batch_id][prompt_id]
                    # ic(batch_label, batch_predict)
                    for i in range(len(batch_label)):
                        whole_dataset[setup_number][prompt_id].append((list(batch_predict[i]), batch_label[i]))
    """
    ic(len(whole_dataset[1]))
    ic(len(whole_dataset[2]))
    ic(len(whole_dataset[3]))
    ic(len(whole_dataset[1][0]))
    ic(len(whole_dataset[3][0]))
    ic(whole_dataset[3][0][0:5])
    """
    ic(len(whole_dataset[1][0]))
    return whole_dataset

def generate_offset(result_list):
    # Return a offset, and evaluate the performance on trainset
    train_list= result_list
    gt_labels = [i[1] for i in train_list]
    predict_prob = np.array([i[0] for i in train_list])
    predict_labels = predict_prob.argmax(axis=1)

    ic(Counter(gt_labels), Counter(predict_labels))
    ic(accuracy_score(gt_labels, predict_labels), f1_score(gt_labels, predict_labels, average='weighted'))

    offset = np.array([0.0,0.0,0.0,0.0,0.0])
    lr = 0.01
    cgt = dict(Counter(gt_labels))
    # Optimization the offsets
    for i in range(10000):
        predict_labels = predict_prob.argmax(axis=1)
        flag = False
        for label in range(5):
            delta = lr*(1-list(predict_labels).count(label)/cgt[label])
            if (delta!=0):
                flag = True
            offset[label]+=delta
            predict_prob[:,label]+=delta
        if (not flag):
            break
    predict_labels = predict_prob.argmax(axis=1)
    print("###########after optimize#############")
    ic(Counter(gt_labels), Counter(predict_labels))
    ic(accuracy_score(gt_labels, predict_labels), f1_score(gt_labels, predict_labels, average='weighted'))
    ic(offset)
    return offset, [accuracy_score(gt_labels, predict_labels), f1_score(gt_labels, predict_labels, average='weighted')]

def evaluate_performance(result_list, offset):
    gt_labels = [i[1] for i in result_list]
    predict_prob = np.array([i[0]+offset for i in result_list])
    #ic(predict_prob[0:10])
    #ic(result_list[0:10])
    predict_labels = predict_prob.argmax(axis=1)
    ic(Counter(gt_labels), Counter(predict_labels))
    ic(accuracy_score(gt_labels, predict_labels), f1_score(gt_labels, predict_labels, average='weighted'))
    return [accuracy_score(gt_labels, predict_labels), f1_score(gt_labels, predict_labels, average='weighted')]

def generate_robustness_KL(result_list, result_list_augment, augmented_dataset, offset):
    from scipy.special import softmax
    def KL(px, py):
        return np.sum(px*np.log(px/py))
    gt_labels = [i[1] for i in result_list]
    predict_prob = np.array([softmax(i[0]+offset) for i in result_list])
    predict_prob_augment = np.array([softmax(i[0]+offset) for i in result_list_augment])
    KL_list = []
    for i in trange(len(result_list_augment)):
        KL_list.append(KL(predict_prob[augmented_dataset[i]['origin_data_idx']], predict_prob_augment[i]))
    #ic(KL_list[0:10])
    ic(np.average(KL_list))
    return np.average(KL_list)

def evaluate_prompt(train_list, dev_list, test_list, augment_list, augmented_dataset):
    ic("Train set performance")
    evals = []
    offset, eval = generate_offset(train_list)
    evals.append(eval)
    ic("Dev set performance")
    eval = evaluate_performance(dev_list, offset)
    evals.append(eval)
    if (test_list == None):
        return evals
    ic("Test set performance")
    eval = evaluate_performance(test_list, offset)
    evals.append(eval)
    if (augment_list == None):
        return evals
    ic("Aug set performance")
    eval = evaluate_performance(augment_list, offset)
    evals.append(eval)
    #ic("Aug set KL_divergence")
    #eval = generate_robustness_KL(test_list, augment_list, augmented_dataset, offset)
    evals.append(eval)
    return evals



if __name__=='__main__':
    from select_evaluation import generate_prompts
    #df = pd.read_csv('prompt_pool.csv')
    #prompt_list = generate_prompts(df)
    #ic(len(prompt_list))
    model_name = 'gpt2-large'
    #subdataset_name = 'test'
    from datasets import load_from_disk
    import datasets
    #base_dataset = load_from_disk("/cluster/project/sachan/zhiheng/causal_prompting/origin_dataset_test")
    #augmented_dataset = load_from_disk("/cluster/project/sachan/zhiheng/causal_prompting/augmented_dataset_test")
    df = pd.read_csv('prompt_pool_augmented.csv')
    prompt_list = generate_prompts(df)
    #out_dataset_path = "/cluster/project/sachan/zhiheng/causal_prompting/gpt2-medium_aligned_dataset_test"
    #align_dataset(model_name, subdataset_name, base_dataset, prompt_list, out_dataset_path)
    train_list = load_whole_result("gpt2-large", 'train')
    dev_list = load_whole_result("gpt2-large", 'dev')
    #ic(len(train_list[2]))
    #test_list = load_whole_result("gpt2-large", 'test')
    #augment_list = load_whole_result("gpt2-large", 'augment_finalize')
    #ic(len(augment_list[1][1]))
    #ic(len(augment_list[2][1]))
    #ic(len(augment_list[3][1]))
    final_result = []
    #generate_offset(augment_list[1][0])
    #final_result.append(evaluate_prompt(train_list[1][0], dev_list[1][0], test_list[1][0], augment_list[1][0], augmented_dataset))
    ic(len(train_list[0]), len(train_list[1]),len(train_list[2]),len(train_list[3]))
    """
    final_result.append(evaluate_prompt(train_list[1][3], dev_list[1][3], None, None, None))
    final_result.append(evaluate_prompt(train_list[1][0], dev_list[1][0], None, None, None))
    final_result.append(evaluate_prompt(train_list[2][1], dev_list[2][1], None, None, None))
    final_result.append(evaluate_prompt(train_list[1][1], dev_list[1][1], None, None, None))
    final_result.append(evaluate_prompt(train_list[2][0], dev_list[2][0], None, None, None))
    final_result.append(evaluate_prompt(train_list[1][2], dev_list[1][2], None, None, None))
    final_result.append(evaluate_prompt(train_list[2][2], dev_list[2][2], None, None, None))
    """

    """
    final_result.append(evaluate_prompt(train_list[1][0], dev_list[1][0], None, None, None))
    final_result.append(evaluate_prompt(train_list[1][1], dev_list[1][1], None, None, None))
    final_result.append(evaluate_prompt(train_list[1][2], dev_list[1][2], None, None, None))
    final_result.append(evaluate_prompt(train_list[1][3], dev_list[1][3], None, None, None))
    final_result.append(evaluate_prompt(train_list[1][4], dev_list[1][4], None, None, None))
    final_result.append(evaluate_prompt(train_list[2][0], dev_list[2][0], None, None, None))
    final_result.append(evaluate_prompt(train_list[2][1], dev_list[2][1], None, None, None))
    final_result.append(evaluate_prompt(train_list[2][2], dev_list[2][2], None, None, None))
    """
    for i in range(5):
        print(len(train_list[i]))
        if (i!=4):
            continue
        for j in range(len(train_list[i])):
            final_result.append(evaluate_prompt(train_list[i][j], dev_list[i][j], None, None, None))

    print(final_result)