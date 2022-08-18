# llm-bivariate-causal-discovery
[[paper](https://openreview.net/forum?id=ucHh-ytUkOH)]
## Introduction
Identifying the causal direction between two variables has long been an important but challenging task for causal inference. Existing work proposes to distinguish whether $X\rightarrow Y$ or $Y \rightarrow X$ by setting up an input-output learning task using the two variables, since causal and anticausal learning have different performances under semi-supervised learning and domain shift. This approach works for many task-specific models trained on the input-output pairs. However, with the rise of general-purpose large language models (LLMs), there are various challenges posed to this previous task-specific learning approach, since continued training of LLMs is less likely to be affordable for university labs, and LLMs are no longer trained on specific input-output pairs. In this work, we propose a new paradigm to distinguish cause from effect using LLMs. Specifically, we conduct post-hoc analysis using natural language prompts that describe different possible causal stories behind the $X$, $Y$ pairs, and test their zero-shot performance. Through the experiments, we show that the natural language prompts that describe the same causal story as the ground-truth data generating direction achieve the highest zero-shot performance, with 2\% margin over anticausal prompts. We highlight that it will be an interesting direction to identify more causal relations using LLMs.

## Code Structure
* `./dataset/`
  * the code about split and augment the dataset
* `./eval/`
  * evaluating the result
* `./model/`
  * warping the model
* `./prompt/`
  * the code about generate the prompt and warping prompt to the dataset
* `./run/`
  * do the inference of gpt2 series model in the given dataset
* `./test/`
  * test codes to the implementation of some modules

## Some special remark
In this version, we use GPT3 API to manually generate the prompt and using `text attack` package to paraphrases the prompt and select the best one. We use the `gpt2` series models with the default in the huggingface.

The origin code is run in the same folder, and to have a more clear structure, the code is split by their functions into 5 folders.
