# [MiLoRA: Harnessing Minor Singular Components for Parameter-Efficient LLM Finetuning](https://arxiv.org/abs/2406.09044)

The official repository containing the introduction and code for our NAACL 2025 paper: [MiLoRA: Harnessing Minor Singular Components for Parameter-Efficient LLM Finetuning](https://arxiv.org/abs/2406.09044).

<p align="center">
  <a href="#-news">| 🔥 News</a> |
  <a href="#-motivation">💡 Motivation</a> |
  <a href="#-milora">🌈 Method</a> |
</p>

<p align="center">
  <!-- <a href="#-experiment-results">| 🔬 Results</a> |
  <a href="#-resources">🪵 Resources</a> | -->
  <a href="#️-quick-start">| ⚡️ Quick Start |</a>
  <a href="#-citation">📓 Citation</a> | 
  <a href="https://aclanthology.org/2025.naacl-long.248/">📃 Paper |</a>

</p>

# 🔥 News
- Jan 2025: Our paper has been accepted by NAACL 2025 main conference.
- Oct 2024: We released our code and quick start.
- May 2024: We released our paper on [arxiv](https://arxiv.org/abs/2406.09044).

# 💡 Motivation
- Full-finetuning is too expensive to train.
- LoRA, the most popular parameter-efficient finetuning method and its varients, are randomly initialized.
- We argue that this strategy may override the important pretrained features, thus degrading the performance of low-rank adaptation methods.

# 🌈 MiLoRA
- To this end, we propose Minor singular component based Low Rank Adaptation (MiLoRA) for efficient LLM finetuning.
- Specifically, we use minor components of the pretrained model to initialize the LoRA.
- This strategy encourages the model to learn in the less-optimized subspace, thus reducing the interference with the well-learned pretrained knowledge encoded in the principal singular components.

<span id="MiLoRA"></span>
![MiLoRA](./assets/imgs/MiLoRA.png)

# ⚡️ Quick Start

## 1. Commonsense Reasoning
We use the code from [LLM-Adapters repo](https://github.com/AGI-Edgerunners/LLM-Adapters) to do commensense reasoning tasks, compare to the lora implement in LLM-Adapters, we only modified it with our LoRA initialization. We diectly use their setting in other experiments without modifications.

## 2. Math Reasoning
Our math reasoning code is modified from [PiSSA](https://github.com/GraphPKU/PiSSA).

### 2.1. Train
For training peft modules on math and code tasks, we do the following preparation.
- Data(download directly in this path):
  - Math dataset: [meta-math/MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA)
  <!-- - Code dataset: [m-a-p/CodeFeedback-Filtered-Instruction](https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction) -->
- Environment:
    ```
    conda create -n milora python=3.10.14 -y
    conda activate milora
    pip install torch==2.3.0
    pip install -r requirements.txt
    ```
- Model(for milora and pissa):
  - ```
    bash scripts/run_svd_init.sh
    ```
    See the shell file and the corresponding .py for more details.

- Train
  
  Run the following shell to train your model, modified `$save_root` to determine which path to save the checkpoints.
  - ```
    # to train milora/pissa/lora

    bash scripts/run_train.sh $method $base_model $save_root

    # e.g. bash scripts/run_train.sh milora ./svd_init_models/LLM-Adapters-rank-64-min ./output

    ```
  See the training log in `./logs`, and we also implement `report_to tensorboard` by default. Use `tensorboard --logdir $save_root` to check tensorboard output.
### 2.2. Evaluation
The GSM8K and MATH evaluations are already included in the training code, check the evaluation results in `results/gsm8k` and  `results/MATH`.


## 3. Instruction Following
We use the implementation in [open-instruct](https://github.com/allenai/open-instruct).

## 4. Visual Instruction tuning
We use the implementation in [DoRA](https://github.com/NVlabs/DoRA/tree/main/visual_instruction_tuning), for hyperparameters, we directly followed the LoRA setting in [Visual Instruction Tuning](https://proceedings.neurips.cc/paper_files/paper/2023/file/6dcf277ea32ce3288914faf369fe6de0-Paper-Conference.pdf).

<!-- # 🔬 Experiment Results
## 1. Commonsense Reasoning
<span id="Commonsense_reasoning_result"></span>
![Commonsense_reasoning_result](./assets/imgs/Commonsense_reasoning_result.png)

## 2. Math Reasoning
<span id="Math_reasoning_result"></span>
![Math_reasoning_result](./assets/imgs/Math_reasoning_result.png)

## 3. Instruction Following
<span id="Instruction_following_result"></span>
![Instruction_following_result](./assets/imgs/Instruction_following_result.png)

## 4. Visual Instruction tuning
<span id="Visual_Instruction_tuning_result"></span>
![Visual_Instruction_tuning_result](./assets/imgs/Visual_Instruction_tuning_result.png)

## 5. Compare with more PEFTs
<span id="Compare"></span>
![Compare](./assets/imgs/Compare.png) -->

<!-- # 🪵 Resources
Our checkpoints will be upload to huggingface. -->

# 📓 Citation
If you find this repo is useful, please cite us as:
```bibtex
@inproceedings{wang-etal-2025-milora,
    title = "{M}i{L}o{RA}: Harnessing Minor Singular Components for Parameter-Efficient {LLM} Finetuning",
    author = "Wang, Hanqing  and
      Li, Yixia  and
      Wang, Shuo  and
      Chen, Guanhua  and
      Chen, Yun",
    editor = "Chiruzzo, Luis  and
      Ritter, Alan  and
      Wang, Lu",
    booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.naacl-long.248/",
    pages = "4823--4836",
    ISBN = "979-8-89176-189-6",
    abstract = "Efficient finetuning of large language models (LLMs) aims to adapt the LLMs with reduced computational and memory costs. Previous LoRA-based approaches initialize the low-rank matrices with Gaussian distribution and zero values while keeping the original weight matrices frozen. However, the trainable model parameters optimized in an unguided subspace might interfere with the well-learned subspace of the pretrained weight matrices. In this paper, we propose MiLoRA, a simple yet effective LLM finetuning approach that only updates the minor singular components of the weight matrix while keeping the principal singular components frozen. It is observed that the minor matrix corresponds to the noisy or long-tail information, while the principal matrix contains important knowledge. The MiLoRA initializes the low-rank matrices within a subspace that is orthogonal to the principal matrix, thus the pretrained knowledge is expected to be well preserved. During finetuning, MiLoRA makes the most use of the less-optimized subspace for learning the labeled dataset. Extensive experiments on commonsense reasoning, math reasoning, instruction following and visual instruction following benchmarks present the superior performance of our method."
}
```
