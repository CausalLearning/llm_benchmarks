# llm_benchmarks

一个用于评估大语言模型（Large Language Models, LLMs）泛化能力、可解释性和可信度的基准评测数据集。


# 目录

- [泛化性](#Generalization)
- [可解释性](#Interpretability)
- [可信度](#Credibility)

# 泛化性（Generalization）
我们使用[MuEP](https://github.com/kanxueli/MuEP)来评估大型语言模型的泛化能力。MuEP 继承了[ALFWorld](https://github.com/alfworld/alfworld) 的原始测试框架，但引入了更大的训练数据集和更细致的评估指标。MuEP的测试集主要通过以下两种方法评估模型的泛化能力：

##### 1. **见过和未见过的测试场景**

> - **见过的场景（Seen）:** 这些场景和模型在训练期间遇到的房间具有类似性，但在对象的位置、数量和视觉外观有所不同。例如，训练期间看到抽屉里的三支红铅笔，而在测试时变为架子上的两支蓝铅笔。
> - **为见过的场景（Unseen:）** 这些是新的任务实例，可能包含已知的对象-容器配对，但始终位于训练期间未见过的房间中，且容器和场景布局有所不同。

见过的场景集旨在衡量分布内的泛化能力，而未见集则衡量分布外的泛化能力。

##### 2. **模板形式的指令形式和自由表达形式的指令**
在 MuEP 中，所有任务的指令都提供模板格式和自由格式两种形式。模板指令遵循固定的句子结构，而自由格式指令则是根据不同人的语言习惯创作的多样化表达。例如以下几个示例：

> (1) 对于pick_and_place_simple类型任务
>
> - 模板指令: "put \<Object> in/on \<Receptacle>"
>
> > ​    - 例如: "put a mug in desk."
>
> - 自由表达形式示例:
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
> (2) 对于pick_heat_then_place_in_recep类型任务
>
> - 模板指令: "cool some \<Object> and put it in \<Receptacle>"
>
> > ​    - 例如: cool some bread and put it in countertop.
>
> - 自由表达形式示例:
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

###### 3.1 模板指令性能评测
<table>
    <tr>
        <td></td>
        <td align="center" colspan="5">Seen(见过的场景)</td>
        <td align="center" colspan="5">Unseen(未见过的场景)</td>
    </tr>
    <tr>
        <td></td>
        <td>SR</td>
        <td>IS</td>
        <td>GCS</td>
        <td>LC</td>
        <td>RDI</td>
        <td>SR</td>
        <td>IS</td>
        <td>GCS</td>
        <td>LC</td>
        <td>RDI</td>
    </tr>
    <tr>
    <td><a href="https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat">Baichuan2-7B</a></td>
    <td>86.43</td>
    <td>11.6</td>
    <td>89.43</td>
    <td>98.22</td>
    <td>5.26</td>
    <td>89.55</td>
    <td>13.78</td>
    <td>92.55</td>
    <td>95.43</td>
    <td>7.14</td>
</tr>
<tr>
    <td><a href="https://huggingface.co/THUDM/chatglm3-6b">ChatGLM3-6B</td>
    <td>86.43</td>
    <td>11.84</td>
    <td>89.22</td>
    <td>100</td>
    <td>31.58</td>
    <td>81.34</td>
    <td>12.88</td>
    <td>84.68</td>
    <td>99.52</td>
    <td>44</td>
</tr>
<tr>
    <td><a href="https://huggingface.co/Qwen/Qwen2-7B-Instruct">Qwen2-7B</td>
    <td>83.57</td>
    <td>12.26</td>
    <td>87.32</td>
    <td>95.36</td>
    <td>0</td>
    <td>86.57</td>
    <td>14.34</td>
    <td>89.14</td>
    <td>98.69</td>
    <td>5.56</td>
</tr>
<tr>
    <td><a href="https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct">LLaMA3-8B</td>
    <td>82.86</td>
    <td>12.08</td>
    <td>87.05</td>
    <td>98.58</td>
    <td>4.17</td>
    <td>85.07</td>
    <td>13.73</td>
    <td>89.65</td>
    <td>97</td>
    <td>10</td>
</tr>
<tr>
    <td><a href="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2">Mistral-7B</td>
    <td>76.43</td>
    <td>12.45</td>
    <td>81.29</td>
    <td>99.38</td>
    <td>3.03</td>
    <td>79.85</td>
    <td>12.35</td>
    <td>83.42</td>
    <td>97.53</td>
    <td>0</td>
</tr>
<tr>
    <td><a href="https://huggingface.co/google/gemma-1.1-7b-it">Gemma-1.1-7b</td>
    <td>79.29</td>
    <td>11.61</td>
    <td>82.76</td>
    <td>98.05</td>
    <td>0</td>
    <td>83.58</td>
    <td>13.1</td>
    <td>87.75</td>
    <td>100</td>
    <td>0</td>
</tr>
</table>

######    3.2 自由表达指令评测
<table>
    <tr>
        <td></td>
        <td align="center" colspan="5">Seen(见过的场景)</td>
        <td align="center" colspan="5">Unseen(未见过的场景)</td>
    </tr>
    <tr>
        <td></td>
        <td>SR</td>
        <td>IS</td>
        <td>GCS</td>
        <td>LC</td>
        <td>RDI</td>
        <td>SR</td>
        <td>IS</td>
        <td>GCS</td>
        <td>LC</td>
        <td>RDI</td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat">Baichuan2-7B</td>
        <td>50</td>
        <td>11.44</td>
        <td>54.43</td>
        <td>92.31</td>
        <td>17.14</td>
        <td>51.49</td>
        <td>13.25</td>
        <td>55.43</td>
        <td>95.31</td>
        <td>12.31</td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/THUDM/chatglm3-6b">ChatGLM3-6B</td>
        <td>40.71</td>
        <td>11.53</td>
        <td>45.46</td>
        <td>96.76</td>
        <td>24.1</td>
        <td>50</td>
        <td>14</td>
        <td>56.2</td>
        <td>99.05</td>
        <td>19.4</td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/Qwen/Qwen2-7B-Instruct">Qwen2-7B</td>
        <td>42.14</td>
        <td>12.32</td>
        <td>46</td>
        <td>95.15</td>
        <td>9.88</td>
        <td>55.22</td>
        <td>15.08</td>
        <td>58.63</td>
        <td>94.67</td>
        <td>6.67</td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct">LLaMA3-8B</td>
        <td>41.43</td>
        <td>11.74</td>
        <td>45.92</td>
        <td>94.35</td>
        <td>10.98</td>
        <td>47.76</td>
        <td>14.27</td>
        <td>51.53</td>
        <td>93.86</td>
        <td>7.14</td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2">Mistral-7B</td>
        <td>34.29</td>
        <td>12.54</td>
        <td>37.46</td>
        <td>90.71</td>
        <td>19.57</td>
        <td>47.01</td>
        <td>12.87</td>
        <td>52.23</td>
        <td>94.41</td>
        <td>16.9</td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/google/gemma-1.1-7b-it">Gemma-1.1-7b</td>
        <td>40.71</td>
        <td>11.75</td>
        <td>45.25</td>
        <td>97.64</td>
        <td>7.23</td>
        <td>52.24</td>
        <td>12.56</td>
        <td>55.65</td>
        <td>97.76</td>
        <td>1.56</td>
    </tr>
</table>
模型泛化性评测更多细节见[benchmarking_generalization](./benchmarking_generalization/README.md)



# 可解释性（Interpretability）

我们使用[Faithful-COT](https://github.com/veronica320/Faithful-COT)来评估大模型的可解释性。思维链（chain-of-thought，COT）作为一种解释大模型内部推理过程的方法，在一定程度上反映了模型的忠实性，即模型内部的行为。Faithful-COT使用了两阶段过程达成模型的忠实推理：
> - **解释推理过程:** 在第一阶段中，不同的模型根据问题与提示模板，生成一系列子问题展示求解过程，即大模型思维链。
> - **求解最终结果:** 在第二阶段中，求解器根据第一阶段生成的子问题求解最终答案，获得忠实的推理结果

在这个过程中，我们使用最终的结果的精确率衡量模型的可解释性，模型的可解释性越好，则其生成的思维链越准确，之后求解器所获得的推理结果精确率越高；若模型的可解释性差，则其生成的推理过程并不符合客观真实的推理过程，导致最终结果的精确率较差。

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
1.我们首先使用了小规模模型进行测试，包括 "Baichuan2-7B", "ChatGLM3-6B", "Qwen2-7B", "LLaMA3-8B", "Mistral-7B", "Gemma-1.1-7b"。在大多数情况下，模型会出现重复示例、超出范围的选项等问题，AQUA数据集的测试示例如下所示：

```
测试用例："question": "Find out which of the following values is the multiple of X, if it is divisible by 9 and 12?\n# Answer option: ['A)36', 'B)15', 'C)17', 'D)5', 'E)7']", "answer": "A", "options": ["A)36", "B)15", "C)17", "D)5", "E)7"], "id": 2}

提示思维链：# Question: In a flight of 600 km, an aircraft was slowed down due to bad weather. Its average speed for the trip was reduced by 200 km/hr and the time of flight increased by 30 minutes. The duration of the flight is:
        # Answer option: ['A)1 hour', 'B)2 hours', 'C)3 hours', 'D)4 hours', 'E)5 hours']
        # Write Python code to solve the following questions. Store your result as a variable named 'answer'.
        # 1. What was the duration of the flight? (independent, support: ["The duration of the flight is"])
        duration = Symbol('duration', positive=True)
        # 2. What is the delay of the flight? (independent, support: ["the time of flight increased by 30 minutes"])
        delay = 30 / 60
        # 3. What was the total flight distance? (independent, support: ["In a flight of 600 km"])
        total_distance = 600
        # 4. What was the original speed? (depends on 1 and 3, support: ["External knowledge: speed is distance over time"])
        original_speed = total_distance / duration
        # 5. What was the reduced speed? (depends on 1, 2, and 3, support: [])
        reduced_speed = total_distance / (duration + delay)
        # 6. What was the duration of the flight if the original speed was 200 km/hr faster than the reduced speed? (depends on 4, 5, and 1, support: [])
        solution = solve_it(original_speed - reduced_speed - 200, duration)
        answer = solution[duration]

Baichuan2-7B：{"id": 2, "answer": "T", "completion": "Here are the solutions for the above questions:\n\n1. In a flight of 600 km, an aircraft was slowed down due to bad weather. Its average speed for the trip was reduced by 200 km/hr ...

ChatGLM3-6B：{"id": 2, "answer": "A", "completion": "\n # Question: In a flight of 600 km, an aircraft was slowed down due to bad weather. Its average speed for the trip was reduced by 200 km/hr ...

Qwen2-7B:{"id": 2, "answer": "N", "completion": "# Question: In a flight of 600 km, an aircraft was slowed down due to bad weather. Its average speed for the trip was reduced by 200 km/hr and the time of flight increased by 30 minutes ...

...
```
测试结果显示小模型无法理解给出的思维链模板,解释性差。


2.在第二阶段，我们使用了上述一些模型的大规模版本，以及其他大规模模型进行测试，包括"Baichuan2-Turbo", "qwen-turbo", "gemini-pro","gpt-3.5-turbo","gpt-4"，测试结果如下所示


| Datase      | Baichuan2-Turbo | qwen-turbo  | gemini-pro  | gpt-3.5-turbo | gpt4        |
| ----------- | --------------- | ----------- | ----------- | ------------- | ----------- |
| AQUA        |     15%         |      8%     |     54%     |      53%      |     83%     |
| sports      |     0%          |     67%     |     98%     |      52%      |     100%    |
| CLUTRR      |     62%         |     57%     |     81%     |      13%      |     72%     |


##### 3. **原始论文**

```
@article{lyu2023faithful,
  title={Faithful chain-of-thought reasoning},
  author={Lyu, Qing and Havaldar, Shreya and Stein, Adam and Zhang, Li and Rao, Delip and Wong, Eric and Apidianaki, Marianna and Callison-Burch, Chris},
  journal={arXiv preprint arXiv:2301.13379},
  year={2023}
}
```



# 可信度（Credibility）
