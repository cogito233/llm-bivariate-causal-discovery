from tqdm import tqdm, trange
from icecream import ic

if __name__=='__main__':
    from datasets import load_from_disk
    dataset = load_from_disk('yelp_review_full')
    ic(dataset)
    dataset['train'] = dataset['train'].shuffle(seed=42)
    sum = [0,0,0,0,0]
    selected_train = []
    selected_dev = []
    ans = 0
    for i in trange(len(dataset['train'])):
        if sum[dataset['train'][i]['label']]<2000:
            sum[dataset['train'][i]['label']]+=1
            selected_train.append(i)
            ans+=1
        elif sum[dataset['train'][i]['label']]<4000:
            sum[dataset['train'][i]['label']]+=1
            selected_dev.append(i)
            ans+=1
        else:
            if (ans==20000):
                break
    dataset['selected_train'] = dataset['train'].select(selected_train)
    dataset['selected_dev'] = dataset['train'].select(selected_dev)
    ic(dataset)
    dataset.save_to_disk('./yelp_review_full_split_train_dev')