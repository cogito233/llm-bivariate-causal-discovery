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



if __name__ == '__main__':

    #parser = argparse.ArgumentParser()
    #parser.add_argument("task_id", help="the selected setup number")
    #args = parser.parse_args()
    #task_id = int(args.task_id)
    #output_path = f"./intermediate_data/augment_test_{task_id}_2.jsonl"
    output_path = f"./intermediate_data/regenerating_dataset.jsonl"
    #ic(task_id, output_path)

    from datasets import load_from_disk

    results = []
    dataset = load_from_disk('./yelp_review_full_split_train_dev')

    dataset_idx = np.load("regenerate_dataset.npy")
    current_point = 0
    process_list = []
    q = Queue()
    for i in range(50):  # 开启5个子进程执行fun1函数
        #begin = (task_id-1)*10000+i*20
        #subdataset = dataset['test'][begin:begin+200]
        subdataset_jsonl = []
        for j in range(1):
            if (current_point>len(dataset_idx)):
                break
            current_idx = int(dataset_idx[current_point])
            datapoint = {
                "idx": current_idx,
                "text": dataset['test'][current_idx]['text'],
                'label': dataset['test'][current_idx]['label']
            }
            current_point += 1
            subdataset_jsonl.append(datapoint)
        p = MyProcess(f"augment_Task_Process{i}", subdataset_jsonl, output_path)  # 实例化进程对象
        p.start()
        time.sleep(1)
        process_list.append(p)

    for i in process_list:
        p.join()
        time.sleep(1)

    """
    with jsonlines.open(output_path, 'a') as writer:
        for i in trange(10000):
            while (True):
                try:
                    result = q.get()
                    writer.write(result)
                    ic(result, sum)
                    break
                except:
                    time.sleep(5)
    """

    """
        process_list = []
        for i in range(5):  #开启5个子进程执行fun1函数
            p = MyProcess('Python') #实例化进程对象
            p.start()
            process_list.append(p)
    
        for i in process_list:
            p.join()
    
        print('结束测试')
    """