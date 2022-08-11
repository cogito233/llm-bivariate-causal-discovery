from warper import TokenizerWarper, generate_prompts

# We want to do is to warp the dataset input text like the gpt2_large
# So we first tokenize by GPT2 tokenlizer, then convert it to the text
class GPT3TokenizerWarper(TokenizerWarper):
    def __init__(self, prompt, tokenizer_path = "./gpt2-large_saved/token"):
        super().__init__(prompt, tokenizer_path)

    def calc_single_str(self, str):
        token_ids = self.encode_single_sentence(str)['input_ids']
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(token_ids))

    def __call__(self, str):
        output_dict = {
            "input_prompts":[],
        }
        if (type(str)==list):
            for i in str:
                output_dict["input_prompts"].append(self.calc_single_str(i))
        else:
            output_dict["input_prompts"].append(self.calc_single_str(str))
        return output_dict
from datasets import load_from_disk
from icecream import ic
if __name__=='__main__':
    import pandas as pd
    df = pd.read_csv('prompt_pool.csv')
    ic(len(df))
    perfix_list, postfix_list = generate_prompts(df, 1)
    ic(len(perfix_list))

    dataset = load_from_disk('./yelp_review_full_split_train_dev')
    subdataset = dataset['train']
    tokenizer_list = []
    for i in range(len(perfix_list)):
        tokenizer_list.append(GPT3TokenizerWarper([perfix_list[i], postfix_list[i]]))
    print(tokenizer_list[0](subdataset[100]['text'])['input_prompts'][0])

