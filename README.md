# Probing the Decision Boundaries of In-context Learning in Large Language Models.

The official code for the paper titled "Probing the Decision Boundaries of In-context Learning in Large Language Models."

---

[arXiv](https://arxiv.org/abs/2406.11233) | [Twitter summary post](https://x.com/siyan_zhao/status/1805277462890492321)

---
**Get LLM decision boundary:**

Example 1: To get the decision boundary of Llama-3-8B on a linear binary classification task with 128 in-context examples per class, run:
```
python get_llm_decision_boundary.py --grid_size=50 --model_name=Llama-3-8B --num_in_context=128 --data_type=linear

```

Expected output:

<img src="https://github.com/siyan-zhao/ICL_decision_boundary/blob/main/Llama-3-8B_128incontext.png" alt="Expected Output" width="600">
---

More code for finetuning coming soon.
