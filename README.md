# Probing the Decision Boundaries of In-context Learning in Large Language Models.
The official code for the paper titled "Probing the Decision Boundaries of In-context Learning in Large Language Models."

[arXiv](https://arxiv.org/abs/2406.11233) | [Twitter summary post](https://x.com/siyan_zhao/status/1805277462890492321)

<p align="center">
  <img src="https://github.com/siyan-zhao/ICL_decision_boundary/raw/main/incontext_num.gif" alt="In-context Learning GIF" width="760">
</p>


---
**Get LLM decision boundary:**

Install packages:

```
pip install -r requirements.txt
```


To get the decision boundary of Llama-3-8B on a linear binary classification task with 128 in-context examples per class, run:
```
python get_llm_decision_boundary.py --grid_size=50 --model_name=Llama-3-8B --num_in_context=128 --data_type=linear

```

Expected output:
<p align="center">
<img src="https://github.com/siyan-zhao/ICL_decision_boundary/blob/main/Llama-3-8B_128incontext.png" alt="Expected Output" width="300">
  </p>
  
---

More code for finetuning coming soon.


---
If you find our work helpful, please consider citing our work:

```
@article{zhao2024probing,
  title={Probing the Decision Boundaries of In-context Learning in Large Language Models},
  author={Zhao, Siyan and Nguyen, Tung and Grover, Aditya},
  journal={arXiv preprint arXiv:2406.11233},
  year={2024}
}
```
