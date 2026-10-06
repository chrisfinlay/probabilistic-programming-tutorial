# Facilitator guide: Introduction to Probabilistic Programming

Big Data Africa School 2026 · 6 October 2026 · Dr Chris Finlay (EPFL)

A one-hour workshop for final-year undergraduates through PhD students (biomedical to computer
science). It covers Bayesian ideas first, then a walkthrough of a hierarchical model in NumPyro,
then Q&A.

## Files

| File | What it is |
|---|---|
| **Slides** | [Slide deck](https://claude.ai/artifact/JKxNK7VEbUc3tkTMzqxfk9): 27 slides with speaker notes, exportable to PDF/PPTX |
| `hierarchical_model_numpyro.ipynb` | Presenter copy, **with outputs**, so it can be shown even if live execution fails. A 15-minute main path, then Extras for self-study |
| `hierarchical_model_numpyro_student.ipynb` | Same notebook with outputs cleared, for participants to run |
| `src/notebook.py` | Source of the notebook (percent format). Edit this, then rebuild (below) |
| `src/build_notebook.py` | Converts `src/notebook.py` into `.ipynb` |
| `requirements.txt` | Tested package versions |

## Setup

**Participants (easiest):** upload `hierarchical_model_numpyro_student.ipynb` to
Google Colab (or host it on GitHub and share an "Open in Colab" link). The first cell installs
NumPyro; JAX is preinstalled on Colab.

**Locally:**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

**Rebuild the notebooks after editing `src/notebook.py`:**

```bash
python src/build_notebook.py src/notebook.py hierarchical_model_numpyro.ipynb
jupyter nbconvert --to notebook --execute --inplace hierarchical_model_numpyro.ipynb
jupyter nbconvert --to notebook --ClearOutputPreprocessor.enabled=True hierarchical_model_numpyro.ipynb --output hierarchical_model_numpyro_student.ipynb
```

The full notebook runs in about 10 seconds on a laptop CPU.

## Run of show (60 min)

| Time | Slides / notebook | Notes |
|---|---|---|
| 0–2 | Slides 1–2 (cover, plan) | Share the notebook link so people can start it running |
| 2–4 | Slide 3 (which hospital is worse?) | Show of hands. Don't answer it yet |
| 4–25 | Slides 4–18 | Belief, probability primer (definitions in symbols, then joint, marginalisation, conditioning on a rain/cloud table), Bayes' rule derived from conditioning, diagnostic test over three slides (named quantities and a guess, Bayes' rule step by step, counting 100,000 people), prior → posterior, priors, MCMC over two slides (why the denominator is hard; how MCMC avoids it), diagnostics, what a PPL is |
| 25–30 | Slides 19–24 (hierarchical models intro, pooling, model, code, workflow, divider) | Introduce the hospital model on slides so the notebook can move quickly |
| 30–45 | Notebook §1–5 (main path) | Stop at "Takeaways". The Extras are for self-study |
| 45–47 | Slides 25–26 (results, takeaways) | Close the loop on Hospital A |
| 47–60 | Slide 27 (Q&A) | See prepared answers below |

**If you're short on time**, cut the counting slide (12; give the 100,000-people version in one
sentence on slide 11), then go straight from the shrinkage plot (§4) to the "worse than typical"
table (§5) without the error table.

### Notebook main path: key moments (15 min)

1. **§1 Data plot (2 min):** the extreme observed rates are 25% (H01, 8 patients) and 0% (H02,
   12 patients). Small hospitals are noisy, not special.
2. **§2 Three models (4 min):** `sample`, `obs=` and `plate`. The only change between the models
   is *where the rate comes from*. The hierarchical model is already in non-centred form (one
   comment explains it; slide 22 covers it).
3. **§3 Fit and check (3 min):** `r_hat` 1.00, healthy `n_eff`, 0 divergences. `tau` ≈ 0.29
   against a true 0.30.
4. **§4 Shrinkage (3 min):** small hospitals get pulled toward the population; large ones barely
   move. Error: raw rates 4.5 pp, complete pooling 2.6, no pooling 2.9, **partial pooling 2.1**.
5. **§5 Answer (3 min):** H01 has P(worse than typical) = 0.61, roughly a coin flip. H09 and H10
   are confidently *better* (P ≈ 0.13–0.14).

**Extras (self-study, after the main path):** A. exact vs MCMC for one hospital (means 0.185 and
0.185); B. prior predictive check (rates from about 3% to 67%); C. centred vs non-centred
(55 vs 0 divergences, with the funnel plot); D. posterior predictive check; E. rank
uncertainty; F. a new hospital's first 50 patients.

(The numbers come from the fixed seeds with the versions in `requirements.txt`. Other
JAX/NumPyro versions may differ slightly; the qualitative story holds.)

## Before the session

- [x] Fill in the workshop, date and presenter (slide 1)
- [x] Add the notebook link (slide 27)
- [ ] Share the deck from its Share menu if participants should see it (it's private by default)
- [ ] Test the student notebook on Colab from a fresh account
- [ ] Have the presenter notebook (with outputs) open as a fallback

## Prepared answers for Q&A

**"Is the diagnostic test (slides 10–12) an appropriate use of Bayes' rule?"**
Yes. It's the textbook example, and statisticians of both schools accept it, because all three
inputs are measurable frequencies: the prior is a base rate, not a subjective belief. (The
contested part of Bayesian statistics comes later, with priors on parameters like `tau`.)
One-line answer: *"Yes, as long as the prior matches the person being tested: 1 in 1,000 is
for screening, and someone with symptoms starts much higher."* Caveats a sharp student may raise:

- **Whose prior?** This matters most. 1 in 1,000 is the prevalence in the *screened population*.
  Someone tested because of symptoms, family history or known exposure has a much higher prior:
  with a prior of 1 in 10, the same positive gives about 92%. The slide's calculation fits
  population screening, not every patient in a clinic.
- **Sensitivity and specificity aren't known exactly.** They're estimated from validation
  studies, which carry uncertainty, and they shift between populations (tests often perform
  differently on mild and severe cases, known as "spectrum bias"). The fully Bayesian approach
  puts priors on them too, which is a natural bridge to the hierarchical model.
- **The second-test update (9% → about 91%) assumes independent errors** given disease status.
  Repeating the *same* assay often repeats the same false positive (e.g. from cross-reactivity),
  which is why confirmatory tests usually use a different method.
- **"99% accurate" is ambiguous.** Overall accuracy depends on prevalence: a test that always
  says "negative" would be 99.9% accurate here. Quote sensitivity and specificity separately,
  as the slides do.
- **9% isn't "probably nothing".** It's about 90 times the baseline risk. Bayes' rule gives the
  probability; the decision (here, a confirmatory test) depends on costs and benefits.

**"Isn't the prior subjective?"**
Every analysis has assumptions; the likelihood is one too. Bayesian analysis writes them down
where reviewers can challenge them. Weakly informative priors rule out absurd values, prior
predictive checks show what they imply, and with enough data the likelihood dominates. You can
also run a sensitivity analysis: refit with different priors and see whether conclusions change.

**"How is this different from a mixed-effects model (lme4, `statsmodels` MixedLM)?"**
Same structure: random intercepts *are* partial pooling. The Bayesian version gives full
posterior uncertainty for every quantity (including `tau`), copes better with few groups or tiny
groups (lme4 often reports a "singular fit" with the variance estimated as exactly 0), and makes
derived questions like ranks or P(worse than typical) trivial to answer.

**"Isn't shrinkage just biasing the estimates?"**
Yes, slightly. It trades a little bias for a large reduction in variance, which lowers the total
error. That's exactly what the error table shows. This is the same idea as the James–Stein estimator
and as regularisation in machine learning.

**"What's the difference between a credible interval and a confidence interval?"**
A 90% credible interval means "given the model and data, there's a 90% probability the parameter
is in here". A confidence interval's 90% describes the long-run behaviour of the procedure over
repeated experiments, not the specific interval. People usually *interpret* confidence intervals
as credible intervals.

**"What do I do if divergences don't go away?"**
In order: reparameterise (non-centred, as here), raise `target_accept_prob` (e.g. 0.95–0.99),
tighten priors that allow absurd values, then question the model itself. Don't ignore them;
divergences mean the posterior was not fully explored.

**"How many chains and samples do I need?"**
Four chains with 1,000 warm-up and 1,000 draws each is a good default. Check `r_hat` ≤ 1.01 and
an effective sample size in the hundreds or more for the quantities you report. Tail quantities
(e.g. 99th percentiles) need more.

**"Stan vs PyMC vs NumPyro — which should I learn?"**
The concepts transfer completely. Stan has the largest applied community and the best
documentation (it's used heavily in biostatistics and the social sciences). PyMC is the most
Pythonic and integrates well with ArviZ. NumPyro is the fastest of the three for many models
thanks to JAX, runs on GPU, and its sibling Pyro connects to deep learning. Pick the one your lab
or field uses.

**"Does this scale to large datasets?"**
NUTS evaluates the full-data gradient at every step, so it gets slow beyond roughly millions of
data points or very many parameters. Options: GPU (NumPyro/JAX), variational inference
(`numpyro.infer.SVI`), which is faster but approximate, or subsampling methods.

**"How do I compare models?"**
Posterior predictive checks first: does the model reproduce the features of the data you care
about? For predictive comparison, use leave-one-out cross-validation (PSIS-LOO) via ArviZ:
`az.from_numpyro(mcmc)`, then `az.loo` / `az.compare`.

**"How does this connect to machine learning?"**
L2 regularisation is a Gaussian prior, and L1 is a Laplace prior. Dropout and ensembles are
approximate Bayesian methods. Bayesian neural networks, variational autoencoders and Gaussian
processes all fit in this framework, and NumPyro/Pyro can express them.

**"How do I report a Bayesian analysis in a paper?"**
State the model and the priors (and why), the software and sampler settings, the convergence
diagnostics (R-hat, ESS, divergences), posterior summaries with intervals, posterior predictive
checks and any prior sensitivity analysis. Kruschke's *Bayesian Analysis Reporting Guidelines*
(Nature Human Behaviour, 2021) is a good checklist.

**"Why model on the log-odds scale?"**
Rates must stay between 0 and 1. On the log-odds scale any real number maps to a valid rate, so
a Normal population distribution and linear predictors (e.g. `+ beta * age`) work naturally. It
is the same link function as logistic regression.

**"What is NUTS actually doing?"**
Hamiltonian Monte Carlo treats the negative log posterior as a landscape and simulates a ball
rolling across it, using gradients to make long moves that stay in high-probability regions. NUTS
(the No-U-Turn Sampler) tunes the trajectory length automatically. Divergences happen when that
simulation breaks down in sharply curved regions like the funnel.

## Further reading

- McElreath, *Statistical Rethinking* (2nd ed.), with lectures on YouTube and a
  [NumPyro port of the code](https://fehiepsi.github.io/rethinking-numpyro/)
- Gelman et al., *Bayesian Data Analysis* (3rd ed.), free PDF from the authors
- Gelman et al. (2020), ["Bayesian Workflow"](https://arxiv.org/abs/2011.01808)
- [NumPyro documentation and examples](https://num.pyro.ai)
- Betancourt, ["A Conceptual Introduction to Hamiltonian Monte Carlo"](https://arxiv.org/abs/1701.02434)
