import numpy as np
import sklearn.metrics
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
from icecream import ic
if __name__=='__main__':
    result_path = "/cluster/project/sachan/zhiheng/causal_prompting/intermediate_data/predict_result.npy"
    a = np.load(result_path, allow_pickle = True)
    result_dict=a.tolist()
    ic(result_dict.keys())
    gt_labels = result_dict['label'][0:2000]
    result_dict['setup1_answer'] = np.array(result_dict['setup1_answer'][0:2000])
    result_dict['setup2_answer'] = np.array(result_dict['setup2_answer'][0:2000])
    result_dict['setup3_answer'] = np.array(result_dict['setup3_answer'][0:2000])
    lr = 0.01
    result1 = np.array(result_dict['setup1_answer']).argmax(axis=1)
    result2 = np.array(result_dict['setup2_answer']).argmax(axis=1)
    result3 = np.array(result_dict['setup3_answer']).argmax(axis=1)
    offset1 = [0,0,0,0,0]
    offset2 = [0,0,0,0,0]
    offset3 = [0,0,0,0,0]
    ic(Counter(gt_labels), Counter(result1),Counter(result2),Counter(result3))
    ic(accuracy_score(gt_labels, result1), f1_score(gt_labels, result1, average='weighted'))
    ic(accuracy_score(gt_labels, result2), f1_score(gt_labels, result2, average='weighted'))
    ic(accuracy_score(gt_labels, result3), f1_score(gt_labels, result3, average='weighted'))
    cgt = dict(Counter(gt_labels))
    for i in range(1000):
        result1 = np.array(result_dict['setup1_answer']).argmax(axis = 1)
        result2 = np.array(result_dict['setup2_answer']).argmax(axis = 1)
        result3 = np.array(result_dict['setup3_answer']).argmax(axis = 1)
        for label in range(5):
            offset1[label]+=lr*(1-list(result1).count(label)/cgt[label])
            offset2[label]+=lr*(1-list(result2).count(label)/cgt[label])
            offset3[label]+=lr*(1-list(result3).count(label)/cgt[label])
            result_dict['setup1_answer'][:,label]+=lr*(1-list(result1).count(label)/cgt[label])
            result_dict['setup2_answer'][:,label]+=lr*(1-list(result2).count(label)/cgt[label])
            result_dict['setup3_answer'][:,label]+=lr*(1-list(result3).count(label)/cgt[label])
    result1 = np.array(result_dict['setup1_answer']).argmax(axis = 1)
    result2 = np.array(result_dict['setup2_answer']).argmax(axis = 1)
    result3 = np.array(result_dict['setup3_answer']).argmax(axis = 1)
    ic(Counter(gt_labels), Counter(result1),Counter(result2),Counter(result3))
    ic(accuracy_score(gt_labels, result1), f1_score(gt_labels, result1, average='weighted'))
    ic(accuracy_score(gt_labels, result2), f1_score(gt_labels, result2, average='weighted'))
    ic(accuracy_score(gt_labels, result3), f1_score(gt_labels, result3, average='weighted'))
    ic(offset1)
    ic(offset2)
    ic(offset3)
    #ic(gt_labels[0:100], result3[0:100])
    #ic(result1[0:100], result2[0:100])