# Faithful-COT
Code and data accompanying our paper ["Faithful Chain-of-Thought Reasoning"](https://arxiv.org/abs/2301.13379) in [IJCNLP-AACL 2023](http://www.ijcnlp-aacl2023.org/).

## Table of Contents
- [Get started](https://github.com/veronica320/Faithful-COT#get-started)
- [Repo Structure](https://github.com/veronica320/Faithful-COT#repo-structure)
- [Usage](https://github.com/veronica320/Faithful-COT#usage)
  - [Make predictions](https://github.com/veronica320/Faithful-COT#make-predictions)
  - [Evaluate model predictions](https://github.com/veronica320/Faithful-COT#evaluate-the-model-predictions)
  - [Get a performance summary table](https://github.com/veronica320/Faithful-COT#get-a-performance-summary-table)
- [Citation](https://github.com/veronica320/Faithful-COT#citation)

## News 📣
- [Nov 2023] 🎉 Our paper won the Area Chair Award (Interpretability and Analysis of Models for NLP) at [IJCNLP-AACL 2023](http://www.ijcnlp-aacl2023.org/)!
- [Sep 2023] Our paper was accepted to [IJCNLP-AACL 2023](http://www.ijcnlp-aacl2023.org/)!
- [Jul 2023] Our paper was presented at the [Natural Language Reasoning and Structured Explanations (NLRSE) Workshop](https://nl-reasoning-workshop.github.io/) (non-archival track) at ACL 2023.
- [Apr 2023] We have added results using **ChatGPT** (`gpt-3.5-turbo`) and **GPT-4** (`gpt-4`) as the underlying Translator LM.

  Here are the accuracy scores in comparison to the original **Codex** (`code-davinci-002`), using the same prompt + greedy decoding:

|         | **GSM8K** | **SVAMP** | **MultiArith** | **ASDiv** | **AQUA** | **saycan** | **StrategyQA** | **date** | **sports** | **CLUTRR** |
|-----------------------------|:---------:|:---------:|:--------------:|:---------:|:--------:|:----------:|:--------------:|:--------:|:----------:|:----------:|
| **Codex** |   72.2    |   83.5    |    **98.8**    |   80.2    |   47.2   | 89.3       |    **63.0**    | 81.6     | 99.1       | 58.9       |
| **ChatGPT**  |   75.8    |   83.0    |      95.3      |   81.7    |   53.5   | 80.6       |      51.5      | 73.5     | 52.3       | 12.1       |
| **GPT-4**     | **95.0**  | **95.3**  |      98.5      | **95.6**  |  **73.6**  | **92.2**       |      54.0      | **95.8**     | **99.3**       | **62.7**       |

With GPT-4, Faithful CoT achieves❗**95.0+** few-shot accuracy❗on almost all Math Word Problem datasets, Date Understanding, and Sports Understanding.

See [`output_dir/performance_summary.csv`](https://github.com/veronica320/Faithful-COT/blob/861aaa9898dfc8e5da7cc5d1e29bbb437f3c9c0f/output_dir/performance_summary.csv) for detailed results and `output_dir/{dataset_name}` for model predictions.


## Get started
We suggest using miniconda/conda to set up the environment. The `environment.yml` file specifies the minimal dependencies. You can create a virtual environment using it according to [this guildeline](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

Essentially, you'll need to do something like:
```
cd /path/to/Faithful-COT
conda env create -f ./environment.yml --prefix ./envs
```

Additionally, to run experiments on **StrategyQA**, you should install [Soufflé](https://souffle-lang.github.io/index.html) (a version of Datalog interpreter we use) following [these instructions](https://souffle-lang.github.io/build). It's not a Python package, so you'll need to install it separately. Note that under the "Installing Souffle" section you should use `-DCMAKE_INSTALL_PREFIX="~/.local"` for it to be installed to the right place. 

## Repo Structure
- `environment.yml`: Conda environment file.
- `data/`: Evaluation datasets. See `data/README.md` for details.
- `output_dir/`: Model prediction files. See `output_dir/README.md` for details.
- `source/`: Source codes.
  - `dataset/`: Dataset utility functions.
  - `configuration/`: model configuration. This is to specify the hyperparameters for each model, such as `"prompt_name"`, `"LM"`, and so on. 
    - `configuration.py`: the Config class. See the definition of each field in the `init()` funciton.
    - `config_files/`: the configuration files for each model. Each file name is in the format of `{model-name}.json`, e.g., `code002_NL+SL.json`. See `configuration/README.md` for details.
  - `prompt/`: The prompts for the LM to generate the reasoning chain. For every prompt with name `{prompt-name}` (e.g., `NL+SL`), there are two files associated with it:
    - `{prompt-name}_prompt.txt`: the prompt containing few-shot examples. See `NL+SL_prompt.txt` for an example.
    - `{prompt-name}_template.txt`: the template to transform a new input example into the desired format. See `NL+SL_template.txt` for an example.
    - The above two files will be used in combination to generate the final prompt to query the LM. See `codex.py` for details.
    - The `{prompt-name}` here corresponds to the `"prompt_name"` field in a configuration file.
  - `model/`:
    - `codex.py`: tur main `Model` class. See inline comments for details.
    - `solver/`: the deterministic solvers for each dataset. They are called by `Model` in executing the reasoning chain to derive the answer.
  - `predict/`:
    - `predict.py`: Run the model to make predictions on a dataset. Output will be saved under `ouput_dir/`.
  - `evaluate/`:
    - `evaluate_answer_acc.py`: Evaluate the answer accuracy for a certain model configuration on a given dataset and split.
    - `gen_perf_table.py`: Generate a table of performance summary for all model configurations on all datasets in `output_dir/`.

## Usage

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

### Get a performance summary table

To reproduce the numbers in our paper, you can generate a performance summary table for all model configurations on all datasets in `output_dir/` by running `source/evaluate/gen_perf_table.py` (no arguments needed).

Example:
```
python gen_perf_table.py
```

The output will be saved to `output_dir/performance_summary.csv`. If there do not exist predictions for a certain model configuration on a dataset, the corresponding cell will be empty.


## Citation
If you find this repository useful, please cite our paper:
```
@article{lyu2023faithful,
  title={Faithful chain-of-thought reasoning},
  author={Lyu, Qing and Havaldar, Shreya and Stein, Adam and Zhang, Li and Rao, Delip and Wong, Eric and Apidianaki, Marianna and Callison-Burch, Chris},
  journal={arXiv preprint arXiv:2301.13379},
  year={2023}
}
```

## Funding Acknowledgements
This research is supported in part by the Office of the Director of National Intelligence (ODNI), Intelligence Advanced Research Projects Activity (IARPA), via the HIATUS Program contract #2022-22072200005. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of ODNI, IARPA, or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for governmental purposes notwithstanding any copyright annotation therein. 






