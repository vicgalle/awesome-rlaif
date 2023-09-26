# awesome-rlaif

A curated and updated list of relevant articles and repositories on **Reinforcement Learning from AI Feedback (RLAIF)**.

## Articles

Articles are sorted chronologically.

#### 2023
- [**2309.00267** RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://arxiv.org/abs/2309.00267)
  - Authors: Harrison Lee, Samrat Phatale, Hassan Mansoor, Kellie Lu, Thomas Mesnard, Colton Bishop, Victor Carbune, Abhinav Rastogi
  - <details> <summary>Abstract</summary> Reinforcement learning from human feedback (RLHF) is effective at aligning large language models (LLMs) to human preferences, but gathering high quality human preference labels is a key bottleneck. We conduct a head-to-head comparison of RLHF vs. RL from AI Feedback (RLAIF) - a technique where preferences are labeled by an off-the-shelf LLM in lieu of humans, and we find that they result in similar improvements. On the task of summarization, human evaluators prefer generations from both RLAIF and RLHF over a baseline supervised fine-tuned model in ~70% of cases. Furthermore, when asked to rate RLAIF vs. RLHF summaries, humans prefer both at equal rates. These results suggest that RLAIF can yield human-level performance, offering a potential solution to the scalability limitations of RLHF. </details>

- [**2308.06385** ZYN: Zero-Shot Reward Models with Yes-No Questions](https://arxiv.org/abs/2308.06385)
  - Authors: Víctor Gallego
  - <details> <summary>Abstract</summary> In this work, we address the problem of directing the text generations of a LLM towards a desired behavior, aligning the generated text with the preferences of the human operator. We propose using another language model as a critic, reward model in a zero-shot way thanks to the prompt of a Yes-No question that represents the user preferences, without requiring further labeled data. This zero-shot reward model provides the learning signal to further fine-tune the base LLM using reinforcement learning, as in RLAIF; yet our approach is also compatible in other contexts such as quality-diversity search. Extensive evidence of the capabilities of the proposed ZYN framework is provided through experiments in different domains related to text generation, including detoxification; optimizing sentiment of movie reviews, or any other attribute; steering the opinion about a particular topic the model may have; and personalizing prompt generators for text-to-image tasks. </details>




#### 2022
- [**2212.08073** Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
  - Authors: Yuntao Bai, Saurav Kadavath, Sandipan Kundu, et al.
  - <details> <summary>Abstract</summary> As AI systems become more capable, we would like to enlist their help to supervise other AIs. We experiment with methods for training a harmless AI assistant through self-improvement, without any human labels identifying harmful outputs. The only human oversight is provided through a list of rules or principles, and so we refer to the method as 'Constitutional AI'. The process involves both a supervised learning and a reinforcement learning phase. In the supervised phase we sample from an initial model, then generate self-critiques and revisions, and then finetune the original model on revised responses. In the RL phase, we sample from the finetuned model, use a model to evaluate which of the two samples is better, and then train a preference model from this dataset of AI preferences. We then train with RL using the preference model as the reward signal, i.e. we use 'RL from AI Feedback' (RLAIF). As a result we are able to train a harmless but non-evasive AI assistant that engages with harmful queries by explaining its objections to them. Both the SL and RL methods can leverage chain-of-thought style reasoning to improve the human-judged performance and transparency of AI decision making. These methods make it possible to control AI behavior more precisely and with far fewer human labels.  </details>



## Code repositories

- [autocrit](https://github.com/CarperAI/autocrit) A repository for transformer critique learning and generation
- [zero-shot-reward-models](https://github.com/vicgalle/zero-shot-reward-models) ZYN: Zero-Shot Reward Models with Yes-No Questions
