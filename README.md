
# GTPO: Trajectory-Based Policy Optimization in Large Language Models

This repository contains the official implementation of **GTPO (Group-relative Trajectory-based Policy Optimization)**, a novel method for stable and effective policy optimization in Large Language Models (LLMs).  
GTPO addresses key limitations of Group-relative Policy Optimization (GRPO), namely:

1. **Token-level gradient conflicts** – where tokens shared across positively and negatively rewarded completions are inconsistently updated, often penalizing essential formatting tokens.
2. **Policy collapse** – where negatively rewarded completions destabilize training, flattening the output distribution and degrading performance.

GTPO introduces **conflict-aware gradient corrections** and **entropy-based regularization** to mitigate these issues, ensuring more stable training without the need for KL-divergence regularization or a reference model.

📄 Paper: [GTPO: Trajectory-Based Policy Optimization in Large Language Models](https://arxiv.org/abs/2508.03772)  
*(Marco Simoni, Aleksandar Fontana, Giulio Rossolini, Andrea Saracino)*

---

## 🚀 Key Contributions

- **Conflict-Aware Gradient Correction**  
  Identifies and protects conflict tokens (appearing in both positively and negatively rewarded completions), preventing harmful penalization while reinforcing beneficial updates.

- **Entropy-Based Policy Regularization**  
  Filters unstable, high-entropy completions and introduces entropy-based penalties to prevent policy collapse and stabilize training.

- **No Reference Model Required**  
  Unlike GRPO, GTPO removes the dependency on KL divergence and external reference policies, reducing computational overhead.

- **Empirical Validation**  
  Extensive experiments on **GSM8K**, **MATH**, and **AIME 2024** benchmarks using **LLaMA-8B** and **Qwen 2.5-3B** demonstrate:
  - More stable training dynamics  
  - Improved reasoning accuracy  
  - Stronger out-of-distribution generalization  

---

## 📊 Results

- GTPO consistently outperforms GRPO and SFT in **accuracy**, **formatting consistency**, and **pass@k/maj@k metrics**.  
- Demonstrates robustness against policy collapse, maintaining performance across long training runs.  
- Out-of-distribution evaluation on **AIME 2024** shows significant improvements in reasoning generalization.  

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/<your-org-or-username>/GTPO.git
cd GTPO
````

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🏋️ Training

Example training script with GTPO:

```bash
python train.py \
  --model llama-8b \
  --dataset gsm8k \
  --optimizer adam \
  --lr 1e-6 \
  --epochs 3 \
  --batch_size 2 \
  --generation_size 12 \
  --use_gtpo
```

Options:

* `--use_grpo` : Train with GRPO baseline
* `--use_sft`  : Train with SFT baseline
* `--generation_size` : Number of completions per group (e.g., 8 or 12)

---

## 📂 Repository Structure

```
GTPO/
│── gtpo/               # Core GTPO implementation
│── scripts/            # Training and evaluation scripts
│── configs/            # Example configurations
│── data/               # Dataset preprocessing utilities
│── results/            # Experimental results and logs
│── requirements.txt    # Dependencies
│── train.py            # Entry point for training
│── eval.py             # Evaluation scripts
└── README.md           # This file
```

---

## 📖 Citation

If you use this code, please cite our paper:

```bibtex
@article{simoni2025gtpo,
  title={GTPO: Trajectory-Based Policy Optimization in Large Language Models},
  author={Simoni, Marco and Fontana, Aleksandar and Rossolini, Giulio and Saracino, Andrea},
  journal={arXiv preprint arXiv:2508.03772},
  year={2025}
}
```

---

## 🤝 Acknowledgements

This work was carried out at:

* **Institute of Informatics and Telematics, CNR, Italy**
* **Department of Excellence in Robotics and AI, TeCIP, Scuola Superiore Sant’Anna**
* **National Doctorate on Artificial Intelligence, Sapienza Università di Roma**

---

## 📬 Contact

For questions or collaborations, please contact:

* Marco Simoni – [marco.simoni@iit.cnr.it](mailto:marco.simoni@iit.cnr.it)
* Aleksandar Fontana – [aleksandar.fontana@santannapisa.it](mailto:aleksandar.fontana@santannapisa.it)
* Giulio Rossolini – [giulio.rossolini@santannapisa.it](mailto:giulio.rossolini@santannapisa.it)
* Andrea Saracino – [andrea.saracino@santannapisa.it](mailto:andrea.saracino@santannapisa.it)

```

