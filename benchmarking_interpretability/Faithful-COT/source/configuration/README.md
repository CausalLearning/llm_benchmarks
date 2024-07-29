# Configuration

This directory contains code and files related to the model configuration, in order to specify the hyperparameters for each model, such as `"prompt_name"`, `"LM"`, and so on. 

The `configuration.py` file contains the Config class. See the definition of each field in the `init()` funciton.

The `config_files/` directory contains model configuration files on all evaluation datasets. Each dataset has a subdirectory. Under it, each configuration file has the name in the format of `{model_name}.json`, e.g., `code002_NL+SL.json`. 
The fields specify hyperparameter values corresponding to those in `source/configuration/configuration.py`.

For all datasets, there are at least the following configurations:
- `code002_NL+SL.json`: Faithful CoT prompt (with both NL comments and SL programs) and the code-davinci-002 LM, under the greedy decoding strategy.
- `code002_NL+SL_n:40.json`: Faithful CoT prompt (with both NL comments and SL programs) and the code-davinci-002 LM, under the self-consistency decoding strategy (number of generations=40).

For the datasets used in Section 6 - Analysis (GSM8K, Saycan, Date Understanding, and CLUTRR), there are additionally the following configurations (all of which are variations based on `code002_NL+SL.json`):
- For Section 6.1 - Ablation Study:
  - `code002_norationale.json`: we remove the rationales, i.e., everything in the brackets from the NL comments, e.g., `"independent, support: [‘There are 15 trees’]"`.
  - `code002_noNLbutnudge.json`: we remove all NL comments except the "nudge" line: e.g., "# To answer this question, we write a Python program to answer the following subquestions".
  - `code002_noNL.json`: we remove all NL comments.
  - `code002_nosolver.json`: instead of calling the external solver, we add "Answer: {answer}" to the end of every exemplar and let the LM predict the answer itself.
- For Section 6.2 - Robustness to Exemplars:
  - `code002_NL+SL_exemplarset1.json`: same prompt format as `code002_NL+SL`, but with a different set of exemplars (set1). 
  - `code002_NL+SL_exemplarset2.json`: same prompt format as `code002_NL+SL`, but with a different set of exemplars (set2). 
  - `code002_NL+SL_exemplarset3.json`: same prompt format as `code002_NL+SL`, but with a different set of exemplars (set3).
  - `code002_NL+SL_exemplarset4.json`: same prompt format as `code002_NL+SL`, but with a different set of exemplars (set4).
  - `code002_NL+SL_exemplarset5.json`: same prompt format as `code002_NL+SL`, but with a different set of exemplars (set5).
- For Appendix C.2 - Effect of LM:
  - `text001_NL+SL.json` (optional): using `text-davinci-001` as the underlying LM. Due to the prompt length limit, text-davinci-001 only allows us to experiment on CLUTRR.
  - `text002_NL+SL.json`: using `text-davinci-002`.
  - `text003_NL+SL.json`: using `text-davinci-003`.
  - `code001_NL+SL.json`: using `code-davinci-001`.
  - `gpt-3.5-turbo_NL+SL.json`(new): using `gpt-3.5-turbo`.
  - `gpt4_NL+SL.json`(new): using `gpt-4`.

- For Appendix C.4 - Robustness to Prompt Phrasing (to be added in the paper): 
  - `code002_prompt_variation1.json`: Randomly permuting the order of independent subquestions/reasoning steps.
  - `code002_prompt_variation2.json`: Changing the variable names.
  - `code002_prompt_variation3.json`: Changing the nudge line (e.g. from “# To answer this question, write a Python program to answer the following subquestions” to “# To solve this question, we answer each of the following subquestions with a Python program”).