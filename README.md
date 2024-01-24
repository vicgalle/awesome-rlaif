# awesome-rlaif ☄️

A curated and updated list of relevant articles and repositories on **Reinforcement Learning from AI Feedback (RLAIF)**. In particular, in this list we keep track of the following motives:

1. Using **RL to optimize LLMs without humans**, i.e., with a critique LM as a reward model.
2. Using LLMs to generate feedback, in a **self-critique** loop.

Some of the listed resources could also be considered as part of RLHF: the frontier is blurry. There are already awesome lists of RLHF, thus, here we make a focus on the previous two points.

## Articles 📚

Articles are sorted chronologically.

#### 2024
- [**2401.10020** Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020)
  - <details> <summary>Abstract</summary> We posit that to achieve superhuman agents, future models require superhuman feedback in order to provide an adequate training signal. Current approaches commonly train reward models from human preferences, which may then be bottlenecked by human performance level, and secondly these separate frozen reward models cannot then learn to improve during LLM training. In this work, we study Self-Rewarding Language Models, where the language model itself is used via LLM-as-a-Judge prompting to provide its own rewards during training. We show that during Iterative DPO training that not only does instruction following ability improve, but also the ability to provide high-quality rewards to itself. Fine-tuning Llama 2 70B on three iterations of our approach yields a model that outperforms many existing systems on the AlpacaEval 2.0 leaderboard, including Claude 2, Gemini Pro, and GPT-4 0613. While only a preliminary study, this work opens the door to the possibility of models that can continually improve in both axes. </details>


#### 2023
- [**2309.00267** RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://arxiv.org/abs/2309.00267)
  - <details> <summary>Abstract</summary> Reinforcement learning from human feedback (RLHF) is effective at aligning large language models (LLMs) to human preferences, but gathering high quality human preference labels is a key bottleneck. We conduct a head-to-head comparison of RLHF vs. RL from AI Feedback (RLAIF) - a technique where preferences are labeled by an off-the-shelf LLM in lieu of humans, and we find that they result in similar improvements. On the task of summarization, human evaluators prefer generations from both RLAIF and RLHF over a baseline supervised fine-tuned model in ~70% of cases. Furthermore, when asked to rate RLAIF vs. RLHF summaries, humans prefer both at equal rates. These results suggest that RLAIF can yield human-level performance, offering a potential solution to the scalability limitations of RLHF. </details>

- [**2309.07124** RAIN: Your Language Models Can Align Themselves without Finetuning](https://arxiv.org/abs/2309.07124)
   - <details> <summary>Abstract</summary> Large language models (LLMs) often demonstrate inconsistencies with human preferences. Previous research gathered human preference data and then aligned the pre-trained models using reinforcement learning or instruction tuning, the so-called finetuning step. In contrast, aligning frozen LLMs without any extra data is more appealing. This work explores the potential of the latter setting. We discover that by integrating self-evaluation and rewind mechanisms, unaligned LLMs can directly produce responses consistent with human preferences via self-boosting. We introduce a novel inference method, Rewindable Auto-regressive INference (RAIN), that allows pre-trained LLMs to evaluate their own generation and use the evaluation results to guide backward rewind and forward generation for AI safety. Notably, RAIN operates without the need of extra data for model alignment and abstains from any training, gradient computation, or parameter updates; during the self-evaluation phase, the model receives guidance on which human preference to align with through a fixed-template prompt, eliminating the need to modify the initial prompt. Experimental results evaluated by GPT-4 and humans demonstrate the effectiveness of RAIN: on the HH dataset, RAIN improves the harmlessness rate of LLaMA 30B over vanilla inference from 82% to 97%, while maintaining the helpfulness rate. Under the leading adversarial attack llm-attacks on Vicuna 33B, RAIN establishes a new defense baseline by reducing the attack success rate from 94% to 19%. </details>

- [**2308.06385** ZYN: Zero-Shot Reward Models with Yes-No Questions](https://arxiv.org/abs/2308.06385)
  - <details> <summary>Abstract</summary> In this work, we address the problem of directing the text generations of a LLM towards a desired behavior, aligning the generated text with the preferences of the human operator. We propose using another language model as a critic, reward model in a zero-shot way thanks to the prompt of a Yes-No question that represents the user preferences, without requiring further labeled data. This zero-shot reward model provides the learning signal to further fine-tune the base LLM using reinforcement learning, as in RLAIF; yet our approach is also compatible in other contexts such as quality-diversity search. Extensive evidence of the capabilities of the proposed ZYN framework is provided through experiments in different domains related to text generation, including detoxification; optimizing sentiment of movie reviews, or any other attribute; steering the opinion about a particular topic the model may have; and personalizing prompt generators for text-to-image tasks. </details>

- [**2307.12950** RLCD: Reinforcement Learning from Contrast Distillation for Language Model Alignment](https://arxiv.org/abs/2307.12950)
  - <details> <summary>Abstract</summary> We propose Reinforcement Learning from Contrast Distillation (RLCD), a method for aligning language models to follow natural language principles without using human feedback. RLCD trains a preference model using simulated preference pairs that contain both a high-quality and low-quality example, generated using contrasting positive and negative prompts. The preference model is then used to improve a base unaligned language model via reinforcement learning. Empirically, RLCD outperforms RLAIF (Bai et al., 2022b) and context distillation (Huang et al., 2022) baselines across three diverse alignment tasks--harmlessness, helpfulness, and story outline generation--and on both 7B and 30B model scales for preference data simulation.</details>


#### 2022
- [**2212.08073** Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
  - <details> <summary>Abstract</summary> As AI systems become more capable, we would like to enlist their help to supervise other AIs. We experiment with methods for training a harmless AI assistant through self-improvement, without any human labels identifying harmful outputs. The only human oversight is provided through a list of rules or principles, and so we refer to the method as 'Constitutional AI'. The process involves both a supervised learning and a reinforcement learning phase. In the supervised phase we sample from an initial model, then generate self-critiques and revisions, and then finetune the original model on revised responses. In the RL phase, we sample from the finetuned model, use a model to evaluate which of the two samples is better, and then train a preference model from this dataset of AI preferences. We then train with RL using the preference model as the reward signal, i.e. we use 'RL from AI Feedback' (RLAIF). As a result we are able to train a harmless but non-evasive AI assistant that engages with harmful queries by explaining its objections to them. Both the SL and RL methods can leverage chain-of-thought style reasoning to improve the human-judged performance and transparency of AI decision making. These methods make it possible to control AI behavior more precisely and with far fewer human labels.  </details>



## Code 🖥️

Here we keep track of repositories and code snippets that are relevant to RLAIF.

- [autocrit](https://github.com/CarperAI/autocrit) A repository for transformer critique learning and generation
- [zero-shot-reward-models](https://github.com/vicgalle/zero-shot-reward-models) ZYN: Zero-Shot Reward Models with Yes-No Questions
- [self-critique chain](https://python.langchain.com/docs/guides/safety/constitutional_chain) Self-critique chain with constitutional AI, using LangChain

## Contributing ❤️

Please, feel free to submit a PR if you want to include resources to this list!
