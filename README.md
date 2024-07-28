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





# Credibility
