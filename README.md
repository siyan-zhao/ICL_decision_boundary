
# Probing the Decision Boundaries of In-context Learning in Large Language Models

This is the official code for the paper titled **"Probing the Decision Boundaries of In-context Learning in Large Language Models."**

[📄 arXiv](https://arxiv.org/abs/2406.11233) | [🧵 Twitter summary post](https://x.com/siyan_zhao/status/1805277462890492321)

<p align="center">
  <img src="https://github.com/siyan-zhao/ICL_decision_boundary/raw/main/incontext_num.gif" alt="In-context Learning GIF" width="760">
</p>

---

## 🔍 Get LLM Decision Boundary

### Installation

Install required packages:

```bash
pip install -r requirements.txt
````

### Example Usage

To get the decision boundary of `Llama-3-8B` on a linear binary classification task with 128 in-context examples per class, run:

```bash
python get_llm_decision_boundary.py --grid_size=50 --model_name=Llama-3-8B --num_in_context=128 --data_type=linear
```

Expected output:

<p align="center">
  <img src="https://github.com/siyan-zhao/ICL_decision_boundary/blob/main/Llama-3-8B_128incontext.png" alt="Expected Output" width="300">
</p>

---

## Finetuning

An example finetuning script on synthetic data is available here:
[`finetune_icl.py`](https://github.com/siyan-zhao/ICL_decision_boundary/blob/main/finetune_icl.py)

For TNP model training code, please refer to:
[https://github.com/tung-nd/TNP-pytorch](https://github.com/tung-nd/TNP-pytorch)

---

## Citation

If you find our work helpful, please consider citing:

```bibtex
@inproceedings{zhao2024probing,
  title={Probing the decision boundaries of in-context learning in large language models},
  author={Zhao, Siyan and Nguyen, Tung and Grover, Aditya},
  booktitle={Proceedings of the 38th International Conference on Neural Information Processing Systems},
  pages={130408--130432},
  year={2024}
}
```

