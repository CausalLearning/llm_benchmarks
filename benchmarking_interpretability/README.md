我们对原始的[Faithful-COT](https://github.com/veronica320/Faithful-COT)进行了精简，删除了对比部分，仅保留了论文中的最终实验设置。在测评的大模型方面，不同于原始论文，我们使用了[llama-factory](https://github.com/hiyouga/LLaMA-Factory)将测评的模型进行了类OpenAI形式的本地部署。

##### 1. **Get started**

## 模型部署



### Make predictions

1. Provide your OpenAI API key(s) by creating a file called `key.py` under `source/` in the following format:
```
API_KEYS = {
	"key1_nickname": "key1",
	"key2_nickname": "key2",
	...
}
```
Note that your keys should have access to the relevant LM (`code-davinci-002`, etc.) specified in the configuration you'd like to use.

2. Choose a model configuration you'd like to use. You can use an existing configuration under `configuration/config_files/{dataset_name}` or create a new one. See `configuration/README.md` for details.

3. Run `source/predict/predict.py`:
```
$ python predict.py -h
usage: predict.py [-h]
                  [--dataset_name {GSM8K,ASDiv,MultiArith,SVAMP,AQUA,date,StrategyQA,sports,saycan,CLUTRR}]
                  [--split {train,dev,test}] [--model_name MODEL_NAME]
                  [--completion_only] [--debug]

optional arguments:
  -h, --help            show this help message and exit
  --dataset_name {GSM8K,ASDiv,MultiArith,SVAMP,AQUA,date,StrategyQA,sports,saycan,CLUTRR}
                        The name of the dataset.
  --split {train,dev,test}
                        The split of the dataset.
  --model_name MODEL_NAME
                        The name of the model (should have a corresponding
                        config file under `configuration/config_files/dataset_name` called
                        `{model_name}.json`.)
  --completion_only     Only query the LM to generate the completion
                        (reasoning chain), but not execute the solver to
                        derive the answer.
  --debug               If true, only run on the first 10 examples.
```


Example:
```
nohup python predict.py --model_name code002_NL+SL --dataset_name GSM8K --split test > logs/GSM8K/code002_NL+SL_test.log 2>&1 &
```

The model predictions will be saved under `output_dir/{dataset_name}/{split}/{model_name}`. See `output_dir/README.md` for details on the format.

Tips: 
- It's recommended to use `nohup` since certain experiments can take hours to run. Also, you may need to create the relevant `logs/{dataset_name}` directory if it doesn't exist.
- The `--completion_only` flag is useful when you run the prediction script on a server, where it may be non-trivial to install certain solvers (e.g. Soufflé). In this case, you can simply run `predict.py` with the `--completion_only` flag on, which will generate the completions only but not derive the answer. Then, on your local machine with the necessary solvers installed, you can run `source/predict/get_answer_from_completion.py` (with the same arguments) to derive the answer from the completions.

### Evaluate the model predictions
Run `source/evaluate/evaluate_answer_acc.py` with the following arguments:
```
$ python evaluate_answer_acc.py -h
usage: evaluate_answer_acc.py [-h]
                              [--dataset_name {GSM8K,ASDiv,MultiArith,SVAMP,AQUA,date,StrategyQA,sports,saycan,CLUTRR}]
                              [--split {train,dev,test}]
                              [--model_name MODEL_NAME] [--non_empty_only]
                              [--valid_only] [--debug]

optional arguments:
  -h, --help            show this help message and exit
  --dataset_name {GSM8K,ASDiv,MultiArith,SVAMP,AQUA,date,StrategyQA,sports,saycan,CLUTRR}
                        The name of the dataset.
  --split {train,dev,test}
                        The split of the dataset.
  --model_name MODEL_NAME
                        The name of the model (should have a corresponding
                        config file under
                        `configuration/config_files/dataset_name` called
                        `{model_name}.json`.)
  --non_empty_only      If true, only evaluate on non-empty answers.
  --valid_only          If true, only evaluate on valid answers.
  --debug               If true, only run on the first 10 examples.
```

The accuracy will be printed to stdout.

Example:
```
python evaluate_answer_acc.py --model_name code002_NL+SL --dataset_name GSM8K --split test
```
Output:
```
Dataset: GSM8K
Split: test
Model: code002_NL+SL
Answer accuracy: 72.2
```