from icecream import ic
with open("seed_prompt.txt") as f:
   content = ''.join(f.readlines())
seed_prompt_candidates = content.split('\n\n<split>\n')
ic(len(seed_prompt_candidates))
df_dict = {
   "text":[],
   "label":[],
}
for i in range(len(seed_prompt_candidates)):
   setup = i//10+1
   df_dict['label'].append(setup)
   if (setup==1):
      text = f"{seed_prompt_candidates[i]}The person’s rating was"
   elif (setup==2):
      text = f"{seed_prompt_candidates[i]}"
   else:
      text = "A person saw this Yelp review: "+"f{review_text}"+f"{seed_prompt_candidates[i]} In this case, the person guessed the rating was "

   df_dict['text'].append(text)

import pandas as pd
ic(len(df_dict['text']))
df = pd.DataFrame.from_dict(df_dict)
df.to_csv("seed_prompt.csv")
"""
[[SETUP 1]]
[For all seed prompts, use them as]

“f{seed_prompt}
The person’s rating was” [now let GPT generate]


[[SETUP 2]]
[For all seed prompts, use them as]

“f{seed_prompt}”

[[SETUP 3]]
[For all seed prompts, use them as]

“A person saw this Yelp review: f{review_text}. f{seed_prompt}. In this case, the person guessed the rating was ”

"""