# Run the examples

Start with the [12-observation conditional-mean example](../methods/model.md#fit-one-conditional-mean-three-ways) to compare PPML, Gamma PML and an intermediate working variance.

Run the scripts from the source checkout or unpacked source distribution
after installing Powermean. The datasets are generated locally
from fixed seeds. The gravity and DiD chapters show the returned tables and
include optional figure code using Matplotlib. To run the figure snippets,
install Matplotlib in the same Python environment.

| Example | Script | What to read |
| --- | --- | --- |
| Gravity with zeros | [`examples/first_fit.py`](../../examples/first_fit.py) | [Three-way PPML](../get-started.md) and [coefficient bias corrections](../methods/bias-correction.md) |
| Firm sales over four years | [`examples/short_panel.py`](../../examples/short_panel.py) | [Firm-cancelled Gamma and counterfactual prediction](../methods/cancellation.md) |
| Nonlinear panel DiD | [`examples/nonlinear_did.py`](../../examples/nonlinear_did.py) | [Common and cohort-period effects](nonlinear-did.md) |
| Two-way exponential regression | [`examples/jochmans.py`](../../examples/jochmans.py) | [Fit GMM1 or GMM2 and predict for panels or directed pairs](../methods/jochmans.md) |
| No-FE PPML with IOLS | [`docs/examples/iols_first_fit.py`](iols_first_fit.py) | [Check the IOLS installation](../iols_installation.md) |

The [sample comparison](compare-samples.md) uses two fits of the same frame
to show retained rows, exclusions, sample summaries and estimates. The
[prepared-fit example](reuse-a-fit.md) reuses a design across solvers,
changes the covariance after estimation and reports a proportional contrast.

<!-- guide-navigation:start -->
[Previous: Compare with ppmlhdfe](../from-ppmlhdfe.md) · [Guide index](../README.md) · [Next: Function index](../reference/api.md)
<!-- guide-navigation:end -->
