# llm_benchmarks

A collection of benchmarks and datasets used to evaluate the generalization, interpretability, and credibility of the LLM. (....)



# Table of Contents

- [Generalization](#Generalization)
- [Interpretability](#Interpretability)
- [Credibility](#Credibility)

# Generalization

We use [MuEP](https://github.com/kanxueli/MuEP) to evaluate the generalization of large language models. MuEP inherits the original testing framework from [ALFWorld](https://github.com/alfworld/alfworld) but incorporates a larger training dataset and fine-grained evaluation metrics. MuEP's testing set primarily assesses model generalization through two methods:

##### 1. **Seen and Unseen Testing Scenarios**

> - **Seen:** This includes known task instances {task-type, object, receptacle, room} in rooms encountered during training, with variations in object locations, quantities, and visual appearances. For example, two blue pencils on a shelf instead of three red pencils in a drawer seen during training.
> - **Unseen:** These are new task instances with potentially known object-receptacle pairs, but always in rooms not seen during training, with different receptacles and scene layouts. 

The seen set is designed to measure in-distribution generalization, whereas the unseen set measures out-of-distribution generalization.

##### 2. **Template and Freedom-form Instruction**

In MuEP, instructions for all tasks are provided in both Template and Free-form formats. The Template instructions follow a fixed sentence structure, while the Free-form instructions are diverse expressions created by humans using their own linguistic habits. Such as the following examples:

> (1) For pick_and_place_simple tasks
>
> - Template Instruction: "put \<Object> in/on \<Receptacle>"
>
> > ​    - Example: "put a mug in desk."
>
> - Freedom-form Instruction Examples:
>
> > ​    - take the mug from the desk shelf to put it on the desk.
> >
> > ​    - Move a mug from the shelf to the desk. 
> >
> > ​    - Move a cup from the top shelf to the edge of the desk.
> >
> > ​    - Transfer the mug from the shelf to the desk surface.
> >
> > ​    - Place the mug on the desk's edge.
>
> (2) For pick_heat_then_place_in_recep tasks
>
> - Template Instruction: "cool some \<Object> and put it in \<Receptacle>"
>
> > ​    - Example: cool some bread and put it in countertop.
>
> - Freedom-form Instruction Examples:
>
> > ​    - Put chilled bread on the counter, right of the fridge.
> >
> > ​    - place the cooled bread down on the kitchen counter
> >
> > ​    - Put a cooled loaf of bread on the counter above the dishwasher.
> >
> > ​    - Let the bread cool and place it on the countertop.
> >
> > ​    - After cooling the bread, set it on the counter next to the stove.



3. ##### Parameter-Efficient Fine-Tuning (PEFT) evaluation result

   ###### 3.1 Template Instruction

|                                                              | Seen  | Seen  |       |        | Seen  | Unseen |       |       |        |       |
| ------------------------------------------------------------ | ----- | ----- | ----- | ------ | ----- | ------ | ----- | ----- | ------ | ----- |
|                                                              | SR    | IS    | GCS   | LC     | RDI   | SR     | IS    | GCS   | LC     | RDI   |
| [Baichuan2-7B](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat) | 86.43 | 11.60 | 89.43 | 98.22  | 5.26  | 89.55  | 13.78 | 92.55 | 95.43  | 7.14  |
| [ChatGLM3-6B](https://huggingface.co/THUDM/chatglm3-6b)      | 86.43 | 11.84 | 86.43 | 100.00 | 31.58 | 81.34  | 12.88 | 81.34 | 99.52  | 44.00 |
| [Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B-Instruct)    | 83.57 | 12.26 | 83.57 | 95.36  | 0.00  |        |       |       |        |       |
| [LLaMA3-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) |       |       |       |        |       |        |       |       |        |       |
| [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) |       |       |       |        |       |        |       |       |        |       |
| [Gemma-1.1-7b](https://huggingface.co/google/gemma-1.1-7b-it) | 79.29 | 11.61 | 79.29 | 98.05  | 0.00  | 83.58  | 13.10 | 83.58 | 100.00 | 0.00  |

######    3.2 Freedom-form Instruction



# Interpretability

我们使用[Faithful-COT](https://github.com/veronica320/Faithful-COT)来评估大模型的可解释性。思维链（chain-of-thought，COT）作为一种解释大模型内部推理过程的方法，在一定程度上反映了模型的忠实性，即模型内部的行为。Faithful-COT使用了两阶段过程达成模型的忠实推理：
> - **解释推理过程:** 在第一阶段中，不同的模型根据问题与提示模板，生成一系列子问题展示求解过程，也即是大模型思维链。
> - **求解最终结果:** 在第二阶段中，求解器根据第一阶段生成的子问题求解最终答案，获得忠实的推理结果

在这个过程中，我们使用最终的结果的精确率衡量模型的可解释性，若模型的可解释性好，则其生成的思维链越准确，之后求解器所获得的推理结果精确率越高；若模型的可解释性差，则其生成的推理过程并不符合客观真实的推理过程，导致最终结果的精确率较差。

##### 1. **推理数据集**
我们按照Faithful-COT原论文，使用了10个评估数据集，其中包括五个数学单词问题（Meth Word Problems，MWP），三个多跳问答数据集（Multi-hop QA），一个规划数据集（Planning）和一个关系推理（Relation inference）数据集。

>MWP: GSM8K (Cobbe et al., 2021), SVAMP (Patel et al., 2021), MultiArith (Roy and Roth, 2015), ASDiv (Miao et al., 2020), and AQuA (Ling et al., 2017)
>>示例："question": "Dan had \$ 3 left with him after he bought a candy bar. If he had $ 4 at the start, how much did the candy bar cost?", "answer": "#### 1"
>
>Multi-hop QA: StrategyQA (Geva et al., 2021), Date Understanding from BIG-bench (BIG-Bench collaboration, 2021), Sports Understanding from BIG-bench
>>示例："Do all parts of the aloe vera plant taste good?","answer":false
>
>Planning: SayCan dataset (Ahn et al.,2022)
>>示例："question": "Visit the table and the counter.", "answer": "[\"1. find(table)\\n2. find(counter)\\n3. done().\\n\"]"
>
>Relation inference: CLUTRR (Sinha et al.,2019) 
>>示例："question": "[Michael] and his wife [Alma] baked a cake for [Jennifer], his daughter.\nQuestion: How is [Jennifer] related to [Alma]?", "answer": "husband-daughter #### daughter", "k": 2

##### 2. **不同模型的评估结果**


##### 3. **原始论文**

```
@article{lyu2023faithful,
  title={Faithful chain-of-thought reasoning},
  author={Lyu, Qing and Havaldar, Shreya and Stein, Adam and Zhang, Li and Rao, Delip and Wong, Eric and Apidianaki, Marianna and Callison-Burch, Chris},
  journal={arXiv preprint arXiv:2301.13379},
  year={2023}
}
```



# Credibility
