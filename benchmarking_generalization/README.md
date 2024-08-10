# 模型泛化性评测

本节将详细介绍大模型泛化性评测的具体细节，包括数据集介绍，评测实施细节和评测结果。



# 目录

- [模型泛化性数据集介绍](#模型泛化性数据集介绍)
- [评测实施细节](#评测实施细节)
- [评测结果](#评测结果)

# 模型泛化性数据集介绍
​	我们使用MuEP来评估大型语言模型的泛化能力。MuEP继承了ALFWorld原始的测试集，但引入了更大的训练数据集和更细致的评估指标。为了对基于基础模型智能体实现更细致的评估，MuEP提供了五种类型的评测指标，包括三种常用的度量成功率、交互步骤和目标条件成功率，以及两种额外设计的评测标准，语言一致性和推理迷失指数。各种指标的具体定义如下：

> - 成功率 (Success Rate，SR)是一个基本指标，表明智能体在完成任务方面的有效性。成功率量化了智能体成功完成的任务数和执行总数的比例,它表示为： SR=S/N，其中S是成功的任务数，N是任务总数。
> - 交互步数 (Interaction Step,IS)衡量智能体的操作效率，它代表了智能体完成任务所需要与环境进行交互的平均步数，它表示为：IS=Step/N，这里的Step表示总的执行步数，N是总任务数。IS整体上反映了智能体的任务规划和执行能力
> - 条件目标成功率 (Goal-Condition Success,GCS)通过计算所有子目标的完成情况来衡量智能体离完全完成任务有多远，计算公式如下：GCS=g/G ，在这个公式中，g是在智能体完成的子目标的数量, G是总的子目标数。
> - 语言一致性 (Language Compliance，LC)旨在衡量基于基础模型的具身智能体输出的动作语言是否符合机器人/智能体控制语法的要求。通过度量模型生成的动作语言的结构化程度和符合性，我们能够准确评估模型在传递有效动作语言方面的能力。LC 的计算公式如下：Lc=V/A这里，V表示执行任务总的有效命令的数量，A表示任务执行的总命令数。更高的LC值表明智能体能输出对齐的结构化语言能力更强。
> - 推理迷失指数 (Reasoning Disorientation Index，RDI)评估智能体在面对任务挑战时陷入无效推理趋势的趋势。具体来说，RDI量化了智能体在面对一致的环境反馈和行动执行场景时诉诸重复行动或迭代、无效解决方案的倾向：RDI=r/f，其中r是RDI计划的次数，f是失败任务的总数。



# 评测实施细节

#### 1. 模型微调与测试框架搭建

##### 1.1 llama_factory环境创建

我们使用[LLaMA_Factory](../common/LLaMA-Factory/)框架来微调和评测所有大模型，故首先需要创建[llama_factory环境创建](../common/LLaMA-Factory/README.md)环境

##### 1.2 添加MuEP数据集信息到llama_factory配置文件中

在[dataset_info.json](../common/LLaMA-Factory/data/dataset_info.json)文件中添加MuEP训练集的信息

##### 1.3 模型微调

进入目录[common\LLaMA-Factory](../common/LLaMA-Factory/)工作目录，执行以下指令启动模型微调web界面，然后选择需要微调的模型和微调配置.

```bash
CUDA_VISIBLE_DEVICES=1 llamafactory-cli webui force_download=True
```

##### 1.4 微调后的模型推理(启动测试)

运行一下指令，启动微调后的模型：

```bash
./inference_alfworld_network.sh  CUDA_NUM Model_name  Lora_Adapter_Path  Template

例如:
./inference_alfworld_network.sh 7 THUDM/chatglm3-6b /home/fist_user2/LLaMA-Factory/saves/ChatGLM3-6B-Chat/lora/train_2024-07-20-20-08-24 chatglm3
```



#### 2.测试环境搭建

##### 2.1 ALFWorld测试环境搭建(安装仿真器)

参考[ALFWorld](https://github.com/alfworld/alfworld)安装仿真器，以及下载相关测试文件.

##### 2.2 进入目录

进入测试客户端目录[MuEP_Benchmark](../benchmarking_generalization/muep_benchmark), 然后运行以下指令

```bash
python network_textworld_client.py
```

##### 2.3 选择测试环境

> - 可以修改network_textworld_client.py的Line 36的URL来请求对应的模型服务器，在Line 144选择seen_valid_path或者unseen_valid_path来选择见过/没见过的环境测试。
> - 在[single_alfworld_tw.py](../benchmarking_generalization/muep_benchmark/single_alfworld_tw.py)的Line 52中设置goal_desc_human_anns_prob=0/1使用模板指令/自由形式指令



# 评测结果

1. ##### 见过和未见过的场景评测

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

2. ##### 固定模板和自由表达形式指令评测

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

2. ALFWorld

> ```bibtex
> @inproceedings{cote2018textworld,
>   title={Textworld: A learning environment for text-based games},
>   author={C{\^o}t{\'e}, Marc-Alexandre and K{\'a}d{\'a}r, {\'A}kos and Yuan, Xingdi and Kybartas, Ben and Barnes, Tavian and Fine, Emery and Moore, James and Hausknecht, Matthew and El Asri, Layla and Adada, Mahmoud and others},
>   booktitle={Workshop on Computer Games},
>   pages={41--75},
>   year={2018},
>   organization={Springer}
> }
> ```

3. MuEP
