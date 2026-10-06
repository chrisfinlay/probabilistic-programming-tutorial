# Introduction to Probabilistic Programming

Big Data Africa School 2026 · 6 October 2026 · Dr Chris Finlay (EPFL)

A hands-on introduction to Bayesian modelling with [NumPyro](https://num.pyro.ai). The notebook
asks a simple question: *twelve hospitals perform the same operation; which ones have worse
complication rates?* It then shows why ranking raw rates misleads, and how a **hierarchical
Bayesian model** gives a better answer.

## Run the notebook

**In your browser (no installation):**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chrisfinlay/probabilistic-programming-tutorial/blob/main/hierarchical_model_numpyro_student.ipynb)

The first cell installs NumPyro; the whole notebook runs in seconds.

**Locally:**

```bash
git clone https://github.com/chrisfinlay/probabilistic-programming-tutorial.git
cd probabilistic-programming-tutorial
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab hierarchical_model_numpyro_student.ipynb
```

## What's here

| File | What it is |
|---|---|
| [`hierarchical_model_numpyro_student.ipynb`](hierarchical_model_numpyro_student.ipynb) | The notebook, ready to run |
| [`hierarchical_model_numpyro.ipynb`](hierarchical_model_numpyro.ipynb) | The same notebook with all outputs, to read without running |
| `src/` | Notebook source and the script that builds it |
| [`FACILITATOR.md`](FACILITATOR.md) | Run of show and prepared Q&A notes |

Slides will be added after the workshop.

## The notebook

**Main path (about 15 minutes):** the data → three models (complete, no and partial pooling) →
fitting and checking with NUTS → shrinkage → which hospitals are really worse than typical?

**Extras for self-study:** Bayes for one hospital (exact vs MCMC), a prior predictive check,
why hierarchical models are written in "non-centred" form, a posterior predictive check,
ranking uncertainty, predicting a new hospital, and exercises.

## Further reading

- McElreath, *Statistical Rethinking* (2nd ed.), with lectures on YouTube and a
  [NumPyro port of the code](https://fehiepsi.github.io/rethinking-numpyro/)
- Gelman et al., *Bayesian Data Analysis* (3rd ed.), free PDF from the authors
- Gelman et al. (2020), ["Bayesian Workflow"](https://arxiv.org/abs/2011.01808)
- [NumPyro documentation and examples](https://num.pyro.ai)
