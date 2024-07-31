# 模型可解释性评测

本节将详细介绍大模型泛化性评测的具体细节，包括数据集介绍，评测实施细节和评测结果。
我们对原始的[Faithful-COT](https://github.com/veronica320/Faithful-COT)进行了精简，删除了对比部分，仅保留了论文中的最终实验设置。在测评的大模型方面，不同于原始论文，我们使用了[llama-factory](https://github.com/hiyouga/LLaMA-Factory)将测评的模型进行了类OpenAI形式的本地部署，使用了API平台进行其他模型的调用。

# 目录

- [模型部署](#模型部署)
- [模型评测](#模型评测)
- [评测结果](#评测结果)


# 模型部署

### 小型模型
在第一阶段的小模型部署中，我们使用了[LLaMA_Factory](../common/LLaMA-Factory/)框架来微调和评测所有大模型，按照[llama_factory环境创建](../common/LLaMA-Factory/README.md)搭建了小规模模型的测试环境。
之后按照OpenAI的标准key形式，通过端口转发进行调用。

### 大型商业模型
我们使用了API平台[柏拉图次元AI](https://api.bltcy.ai/)，通过替换请求网址，使用其给出的专属key，直接调用平台中的商业模型。

# 模型评测

评测方式与原始代码中的调用方式相同，实施细节请参考[Faithful-COT模型预测](Faithful-COT/README.md)。不同点在于替换OpenAI接口的请求地址，和API key
> 小模型调用：
>>请求地址：服务器地址：转发端口/v1
>
>>api key: API_KEYS = {	"key_name": "0"}
>
>商业模型调用：
>>请求地址：api.bltcy.ai/v1
>
>>api key: API_KEYS = {	"key_name": "平台key"}

数据集介绍请参考[Faithful-COT数据集](Faithful-COT/data/README.md)
提问和评测过程请参考[Faithful-COT模型预测](Faithful-COT/README.md)

# 评测结果
1.我们首先使用了小规模模型进行测试，包括 "Baichuan2-7B", "ChatGLM3-6B", "Qwen2-7B", "LLaMA3-8B", "Mistral-7B", "Gemma-1.1-7b"。在大多数情况下，模型会出现重复示例、超出范围的选项等问题，AQUA数据集的测试示例如下所示：


>测试用例：
>
>>"question": "Find out which of the following values is the multiple of X, if it is divisible by 9 and 12?\n# Answer option: ['A)36', 'B)15', 'C)17', 'D)5', 'E)7']", "answer": "A", "options": ["A)36", "B)15", "C)17", "D)5", "E)7"], "id": 2}
>
>提示思维链：以如下形式思考当前问题
>
>> Question: In a flight of 600 km, an aircraft was slowed down due to bad weather. Its average speed for the trip was reduced by 200 km/hr and the time of flight increased by 30 minutes. The duration of the flight is:
>>        # Answer option: ['A)1 hour', 'B)2 hours', 'C)3 hours', 'D)4 hours', 'E)5 hours']
>>        # Write Python code to solve the following questions. Store your result as a variable named 'answer'.
>
>        # 1. What was the duration of the flight? (independent, support: ["The duration of the flight is"])
>        duration = Symbol('duration', positive=True)
>
>        # 2. What is the delay of the flight? (independent, support: ["the time of flight increased by 30 minutes"])
>        delay = 30 / 60
>
>        # 3. What was the total flight distance? (independent, support: ["In a flight of 600 km"])
>        total_distance = 600
>
>        # 4. What was the original speed? (depends on 1 and 3, support: ["External knowledge: speed is distance over time"])
>        original_speed = total_distance / duration
>
>        # 5. What was the reduced speed? (depends on 1, 2, and 3, support: [])
>        reduced_speed = total_distance / (duration + delay)
>
>        # 6. What was the duration of the flight if the original speed was 200 km/hr faster than the reduced speed? (depends on 4, 5, and 1, support: [])
>        solution = solve_it(original_speed - reduced_speed - 200, duration)
>        answer = solution[duration]
>
>回答：
>> - Baichuan2-7B：{"id": 2, "answer": "T", "completion": "Here are the solutions for the above questions:\n\n1. In a flight of 600 km, an aircraft was slowed down due to bad weather. Its average speed for the trip was reduced by 200 km/hr ...
>
>> - ChatGLM3-6B：{"id": 2, "answer": "A", "completion": "\n # Question: In a flight of 600 km, an aircraft was slowed down due to bad weather. Its average speed for the trip was reduced by 200 km/hr ...
>
>> - Qwen2-7B:{"id": 2, "answer": "N", "completion": "# Question: In a flight of 600 km, an aircraft was slowed down due to bad weather. Its average speed for the trip was reduced by 200 km/hr and the time of flight increased by 30 minutes ...
>
>>...
>
测试结果显示小模型无法理解给出的思维链模板,解释性差。


2.在第二阶段，我们使用了上述一些模型的大规模版本，以及其他大规模商业模型进行测试，包括"Baichuan2-Turbo", "qwen-turbo", "gemini-pro","gpt-3.5-turbo","gpt-4"，测试结果如下所示


| Datase      | Baichuan2-Turbo | qwen-turbo  | gemini-pro  | gpt-3.5-turbo | gpt4        |
| ----------- | --------------- | ----------- | ----------- | ------------- | ----------- |
| AQUA        |     15%         |      8%     |     54%     |      53%      |     83%     |
| sports      |     0%          |     67%     |     98%     |      52%      |     100%    |
| CLUTRR      |     62%         |     57%     |     81%     |      13%      |     72%     |

# 引用

1. LLaMA_Factory

> ```bibtex
> @inproceedings{zheng2024llamafactory,
>   title={LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models},
>   author={Yaowei Zheng and Richong Zhang and Junhao Zhang and Yanhan Ye and Zheyan Luo and Zhangchi Feng and Yongqiang Ma},
>   booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)},
>   address={Bangkok, Thailand},
>   publisher={Association for Computational Linguistics},
>   year={2024},
>   url={http://arxiv.org/abs/2403.13372}
> }
> ```
>

2. Faithful-COT

>```bibtex
> @article{lyu2023faithful,
>  title={Faithful chain-of-thought reasoning},
>  author={Lyu, Qing and Havaldar, Shreya and Stein, Adam and Zhang, Li and Rao, Delip and Wong, Eric and Apidianaki, Marianna and Callison-Burch, Chris},
>  journal={arXiv preprint arXiv:2301.13379},
>  year={2023}
>}
>```