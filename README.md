# Powermean for Python <img src="docs/assets/powermean-logo.png" align="right" width="120" alt="Powermean logo" />

How much does an offer change the expected number of orders, including customers who order nothing? Powermean fits regressions for questions like this and reports proportional changes with confidence intervals. You choose which conditional mean to model and how to weight observations. The same formula can include several fixed effects. [Benatia, Bellégo and Pape (2026)](https://arxiv.org/abs/2203.11820v4)

[User guide](docs/README.md) · [Function index](docs/reference/api.md) · [Troubleshooting](docs/help/troubleshooting.md) · [Powermean for R](https://github.com/dvbn/powermean-R)

## Installation

Install from PyPI with Python 3.12 or later. [Installation and platform requirements](docs/install.md)

```bash
python -m pip install "powermean>=0.1.1"
```

## A first fit

Simulate 2,000 customers, give each a 50% chance of receiving a promotion, and fit the arithmetic mean of orders by Poisson pseudo-maximum likelihood (PPML).

```python
import numpy as np
import pandas as pd
import powermean as pm

rng = np.random.default_rng(20261123)
offer = rng.binomial(1, 0.5, size=2000)
customers = pd.DataFrame({
    "offer": offer,
    "orders": rng.poisson(np.exp(0.2 + 0.4 * offer)),
})

fit = pm.fit("orders ~ offer", customers, omega=1, tau=1)
change = fit.effect("offer", contrast=(0, 1))
print(f"{100 * change.estimate:.1f}%")
# 53.5%
```

The fitted mean number of orders is **53.5% higher** for customers who receive the offer (95% interval, 42.6% to 65.2%). All 478 customers who placed zero orders contribute. `change.estimate` and `change.interval` are on the proportion scale.

```python
profiles = pd.DataFrame({"offer": [0, 1]})
print(fit.predict(profiles).round(2).tolist())
# [1.16, 1.78]
```

The model predicts 1.16 orders without the offer and 1.78 with it. Use `fit.predict(type="residual")` for observed minus fitted orders. With fixed effects, `fit.fixef()` returns estimates by level. You can change covariates on the retained rows and predict with those effects held fixed. [Worked example with firm and year effects](docs/reference/results.md#predict-from-a-cancelled-gamma-fit)

## Where to go next

| If you want to… | Read |
| --- | --- |
| Choose between the arithmetic mean, another power mean, and log regression | [Choose the mean order](docs/methods/model.md#choose-a-conditional-mean) |
| Compare PPML and Gamma PML for the same mean | [One conditional mean, three working variances](docs/methods/model.md#fit-one-conditional-mean-three-ways) |
| Specify the sample, weights, exposure or an offset | [Prepare the data](docs/help/prepare-data.md) |
| Check the estimation sample or export a regression table | [Compare fitted rows](docs/examples/compare-samples.md) · [Report and export results](docs/examples/reuse-a-fit.md) |
| Fit trade in levels with pair and country-year effects | [Gravity walkthrough](docs/get-started.md) |
| Estimate effects by cohort and date | [Nonlinear difference-in-differences](docs/examples/nonlinear-did.md) |
| Cancel a fixed effect or correct short-panel bias | [Cancel one block](docs/methods/cancellation.md) · [Compare bias corrections](docs/methods/bias-correction.md) |

## Citation and license

Cite Benatia, Bellégo and Pape (2026) and record your package version. The [papers and methods guide](docs/references.md) lists the references for individual estimators. You can also download the [HTML guide](docs/guide.html) for offline use.

Powermean's own source is licensed under [BSD-3-Clause](LICENSE).
Bundled components carry the [scikit-learn notice](src/powermean/_vendor/SKLEARN_COPYING) and [native extension notices](licenses/NATIVE_NOTICE.md).
