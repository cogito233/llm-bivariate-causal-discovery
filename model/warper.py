import torch
from torch import nn
from transformers import AutoTokenizer
from transformers import GPT2LMHeadModel
import sys
from icecream import ic
sys.path.append('/cluster/project/sachan/zhiheng/causal_prompting')

def generate_prompts(df, setup):
    perfix = []
    postfix = []
    for i in range(len(df)):
        if (int(df['group'].iloc[i])!=setup and setup!=0):
            continue
        current_str = df['text'].iloc[i].replace("\\n","\n")
        perfix.append(current_str.split(" f{review_text} ")[0])
        postfix.append(current_str.split(" f{review_text} ")[-1])
    ic(len(perfix))
    print(perfix)
    print(postfix)
    return perfix, postfix

class ModelWarper(nn.Module):
    def __init__(self, offset = [0,0,0,0,0], model_path = './gpt2-large_saved/model'):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.device = 'cpu'
        if (torch.cuda.is_available()):
            self.device = 'cuda:0'
        self.model = self.model.to(self.device)
        self.label_idx = torch.tensor([352, 362, 513, 604, 642])
        #self.offset = torch.tensor(offset).to(self.device)

    def forward(self, **kwargs):
        for i in kwargs:
            kwargs[i] = kwargs[i].to(self.device)
            #ic(kwargs[i].shape)
        output = self.model(**kwargs)[0][:, :, self.label_idx]
        # ic(output)
        answer = []
        for i in range(len(kwargs['input_ids'])):
            selected_output = output[i][kwargs['attention_mask'][i]== 1][-1].cpu().detach().numpy()
            answer.append(selected_output)
        #ic(answer)
        return answer#+self.offset

# Use the maxlength method to warp the sentence
class TokenizerWarper(object):
    def __init__(self, prompt, tokenizer_path = "./gpt2-large_saved/token"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, truncation = True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.prompt = prompt
        """
        self.perfix_token = self.tokenizer(prompt[0]+"\"")
        self.postfix_token = self.tokenizer("\" "+prompt[1])
        self.postfix_token_2 = self.tokenizer("...\" "+prompt[1])
        """

        self.perfix_token = self.tokenizer(prompt[0])
        self.postfix_token = self.tokenizer(" "+prompt[1])
        self.postfix_token_2 = self.tokenizer("... "+prompt[1])
        self.max_length = 1024

    def padding(self, padded_dict, max_length):
        if (len(padded_dict['input_ids'])>=max_length):
            #ic(len(padded_dict['input_ids']))
            pass
        for i in padded_dict:
            padded_dict[i] = padded_dict[i] + [0]*(max_length-len(padded_dict[i]))
        return padded_dict

    def encode_single_sentence(self, str):
        #str = '\n'.join(str.split('\\n'))
        mid = self.tokenizer(str)
        perfix_token = self.perfix_token
        postfix_token = self.postfix_token
        postfix_token_2 = self.postfix_token_2
        if (len(perfix_token['input_ids'])+len(mid['input_ids'])+len(postfix_token['input_ids'])>self.max_length):
            #ic(len(perfix_token['input_ids'])+len(mid['input_ids'])+len(postfix_token['input_ids']))
            pass
        if (len(perfix_token['input_ids'])+len(mid['input_ids'])+len(postfix_token['input_ids'])>self.max_length):
            return {
                'input_ids':perfix_token['input_ids']+mid['input_ids'][:self.max_length-len(perfix_token['input_ids'])-len(postfix_token_2['input_ids'])]+postfix_token_2['input_ids'],
                'attention_mask':perfix_token['attention_mask']+mid['attention_mask'][:self.max_length-len(perfix_token['attention_mask'])-len(postfix_token_2['attention_mask'])]+postfix_token_2['attention_mask'],
            }
        else:
            return {
                'input_ids':perfix_token['input_ids']+mid['input_ids']+postfix_token['input_ids'],
                'attention_mask':perfix_token['attention_mask']+mid['attention_mask']+postfix_token['attention_mask'],
            }

    def encode(self, str):
        answer = {
            "input_ids":[],
            "attention_mask":[],
        }
        if (type(str)==list):
            max_length = 0
            answers = []
            for i in str:
                answers.append(self.encode_single_sentence(i))
                max_length = max(max_length, len(answers[-1]['input_ids']))
            max_length = min(max_length, self.max_length)
            for subanswer in answers:
                paded_subanswer = self.padding(subanswer, max_length)
                answer['input_ids'].append(subanswer['input_ids'])
                answer['attention_mask'].append(subanswer['attention_mask'])
            ic(max_length)
        else:
            subanswer = self.encode_single_sentence(str)
            answer['input_ids'].append(subanswer['input_ids'])
            answer['attention_mask'].append(subanswer['attention_mask'])
            return answer
        return {'input_ids': torch.tensor(answer['input_ids']), 'attention_mask': torch.tensor(answer['attention_mask'])}

    def __call__(self, str):
        return self.encode(str)

tokenizer = TokenizerWarper(["I just finished eating at a restaurant. Then I opened my Yelp app.\n\nI first gave a rating, and then wrote the following review:\n\n",
                                 "\n\nThe review is an explanation of why I gave a rating (out of 1 to 5 stars) of"])

"""
perfix_list = ["I just finished eating at a restaurant. Then I opened my Yelp app.\n\nI first gave a rating, and then wrote the following review:\n\n",
                   "I just finished eating at a restaurant. Then I opened my Yelp app, and wrote the following review: \n\n",
                   "I opened my Yelp app, and started to read some reviews of a restaurant that I want to try. I saw a user wrote this review:\n\n"]
postfix_list = ["\n\nThe review is an explanation of why I gave a rating (out of 1 to 5 stars) of",
                    "\n\nThen I gave the rating. In terms of 1 to 5 stars, I think this restaurant is worth a",
                    "\n\nIn terms of 1 to 5 stars, I think this user rated it a"]
tokenizer =            
"""
"""
model_gpt = ModelWarper(offset=[-0.35403400000000423,
               0.8431049999999987,
               0.10486800000000042,
               -0.3882869999999973,
               -0.20565200000000122])
import textattack
import textattack.models
import textattack.models.wrappers
from text_attack_pytorch_warper import PyTorchModelWrapper as tmw
model = tmw(model_gpt, tokenizer)
from icecream import ic
ic(model)
"""

def generate_final_dict(model, tokenizer, dataset):
    final_dict = {}
    pass

if __name__=='__main__':

    from datasets import load_from_disk
    dataset = load_from_disk('./yelp_review_full_split_train_dev')

    ic(dataset)
    ic(dataset['train'][0:1]['text'])
    tokenizer = TokenizerWarper(["I just finished eating at a restaurant. Then I opened my Yelp app.\n\nI first gave a rating, and then wrote the following review:\n\n",
                                 "\n\nThe review is an explanation of why I gave a rating (out of 1 to 5 stars) of"])
    tokens = tokenizer.encode(dataset['train'][0:1]['text'])
    ic(tokens)
    model = ModelWarper(offset=[-0.35403400000000423,
               0.8431049999999987,
               0.10486800000000042,
               -0.3882869999999973,
               -0.20565200000000122])
    ic(model(**tokens))


