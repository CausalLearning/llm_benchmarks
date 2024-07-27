# llm_benchmarks
A collection of benchmarks and datasets used to evaluate the generalization, interpretability, and credibility of the LLM.

# Generalization
We use MuEP (URL) to evaluate the generalization of large language models. MuEP inherits the original testing framework from ALFWorld (URL) but incorporates a larger training dataset and fine-grained evaluation metrics. MuEP's testing set primarily assesses model generalization through two methods:
（1）Seen and Unseen Testing Scenarios
-**Seen:** This includes known task instances {task-type, object, receptacle, room} in rooms encountered during training, with variations in object locations, quantities, and visual appearances. For example, two blue pencils on a shelf instead of three red pencils in a drawer seen during training.
-**Unseen:** These are new task instances with potentially known object-receptacle pairs, but always in rooms not seen during training, with different receptacles and scene layouts. 
The seen set is designed to measure in-distribution generalization, whereas the unseen set measures out-of-distribution generalization.

（2）Template and Freedom-form Instruction
Such as following examples:
(1) For pick_and_place_simple tasks
-Template Instruction: "put <Object> in/on <Receptacle>"
---Example: "put a mug in desk."
-Freedom-form Instruction Examples:
---take the mug from the desk shelf to put it on the desk.
---Move a mug from the shelf to the desk. 
---Move a cup from the top shelf to the edge of the desk.
---Transfer the mug from the shelf to the desk surface.
---Place the mug on the desk's edge.

(2) For pick_heat_then_place_in_recep tasks
-Template Instruction: cool some <Object> and put it in <Receptacle>
---Example: cool some bread and put it in countertop.
-Freedom-form Instruction Examples:
---Put chilled bread on the counter, right of the fridge.
---place the cooled bread down on the kitchen counter
---Put a cooled loaf of bread on the counter above the dishwasher.
---Let the bread cool and place it on the countertop.
---After cooling the bread, set it on the counter next to the stove.

# Interpretability

# Credibility
