# Compare estimation samples

Two regressions can report the same number of observations and use different
rows. Check the sample before interpreting a difference between coefficients.
This example fits two models to the same input frame and follows the retained
rows from the result to the descriptive and regression tables.

## Fit both models on the same frame

Suppose you model weekly visits using a promotion indicator. One
specification adjusts for price, and the other adjusts for stock.
A price is missing in week 5 and a stock value is missing in week 2.
Use the same frame for both models.

```python
import pandas as pd
import powermean as pm

data = pd.DataFrame(
    {
        "visits":    [2, 0, 9, 5, 4, 10, 7, 6, 0, 2, 11, 3],
        "promotion": [0, 0, 1, 0, 0,  1, 1, 0, 0, 0,  1, 1],
        "price":     [8, 7, 6, 7, None, 6, 7, 8, 9, 9, 5, 8],
        "stock":     [21, None, 35, 28, 17, 33, 25, 30, 20, 22, 32, 24],
    },
    index=pd.Index([f"week-{i:02d}" for i in range(1, 13)], name="week"),
)
with_price = pm.fit("visits ~ 1 + promotion + price", data, omega=1, tau=1)
with_stock = pm.fit("visits ~ 1 + promotion + stock", data, omega=1, tau=1)
```

## Find the rows each fit used

```python
print("Adjust for price", *with_price.sample.table(), sep="\n")
print("Adjust for stock", *with_stock.sample.table(), sep="\n")
print(with_price.sample.positions)
print(with_stock.sample.positions)
print(with_price.sample.labels)
print(with_stock.sample.labels)
print(with_price.sample.reason(4))
print(with_stock.sample.reason(1))
```

```text
Adjust for price
('subset', 0)
('missing', 1)
('missing: outcome', 0)
('missing: regressors', 1)
('missing: effects', 0)
('missing: instruments', 0)
('missing: weights', 0)
('missing: offset', 0)
('missing: clusters', 0)
('zero_outcome', 0)
('zero_weight', 0)
('retained', 11)
Adjust for stock
('subset', 0)
('missing', 1)
('missing: outcome', 0)
('missing: regressors', 1)
('missing: effects', 0)
('missing: instruments', 0)
('missing: weights', 0)
('missing: offset', 0)
('missing: clusters', 0)
('zero_outcome', 0)
('zero_weight', 0)
('retained', 11)
[ 0  1  2  3  5  6  7  8  9 10 11]
[ 0  2  3  4  5  6  7  8  9 10 11]
Index(['week-01', 'week-02', 'week-03', 'week-04', 'week-06', 'week-07',
       'week-08', 'week-09', 'week-10', 'week-11', 'week-12'],
      dtype='str', name='week')
Index(['week-01', 'week-03', 'week-04', 'week-05', 'week-06', 'week-07',
       'week-08', 'week-09', 'week-10', 'week-11', 'week-12'],
      dtype='str', name='week')
missing
missing
```

Compare the exact retained positions before comparing coefficients. Both fits
read the same input frame in the same order, so use
`same_rows = with_price.sample.positions.tolist() == with_stock.sample.positions.tolist()`.
Here `same_rows` is `False`, even though both fits retain 11 of 12 rows.

`sample.table()` gives the counts in the order Powermean applies the sample
rules. Both fits have one row under `"missing: regressors"`.
`with_price.sample.reason(4)` and `with_stock.sample.reason(1)` each return
`"missing"`, the first recorded reason. Open those zero-based input positions
with `data.iloc[[4, 1]]` to see the missing price in `week-05` and missing stock
in `week-02`. Use `sample.labels` to display the original index labels.
For a retained position, `sample.reason(position)` returns `None`.
[Sample record](../reference/results.md#inspect-the-estimation-sample)

## Describe the two retained samples

```python
price_sample = pm.summary_statistics(
    data, vars=["visits"], probs=(0.5,), sample=with_price,
)
stock_sample = pm.summary_statistics(
    data, vars=["visits"], probs=(0.5,), sample=with_stock,
)
print(price_sample)
```

```text
Summary statistics                          N = 11 rows
-------------------------------------------------------
         N  missing  mean    sd   p50  zeros  mean | >0
-------------------------------------------------------
visits  11        0  5.00  3.92  5.00  18.2%       6.11
-------------------------------------------------------
Rows: the estimation sample of the fit (11 of 12).
```

For the stock-adjusted fit, use the same variables and original frame.

```python
print(stock_sample)
```

```text
Summary statistics                          N = 11 rows
-------------------------------------------------------
         N  missing  mean    sd   p50  zeros  mean | >0
-------------------------------------------------------
visits  11        0  5.36  3.59  5.00   9.1%       5.90
-------------------------------------------------------
Rows: the estimation sample of the fit (11 of 12).
```

The price model keeps both zero-visit weeks and has a mean of 5.00
visits. The stock model omits one zero-visit week and has a mean of
5.36. The outcome summaries differ despite the equal number of rows.

Pass the original frame in its original order with `sample=fit` to any of the
four descriptive table functions. Each uses the fit's retained positions,
checks the frame's row count and index, and applies its own missing-value
rules. [Descriptive table calls](../reference/tables.md#describe-data-and-treatment-arms)

### Summarize outcomes by promotion within one sample

```python
groups = pm.outcome_table(
    data, outcomes="visits", treatment="promotion", sample=with_price,
)
print(groups)
```

```text
Outcome table                                                               control: promotion = 0 (N = 6); promotion = 1 (N = 5)
---------------------------------------------------------------------------------------------------------------------------------
        target   zeros (0)  zeros (1)  mean (0)  mean (1)  mean | >0 (0)  mean | >0 (1)  difference, unadjusted  [95% conf. int.]
---------------------------------------------------------------------------------------------------------------------------------
visits  theta_N      33.3%       0.0%      2.50      8.00           3.75           8.00                  220.0%   [38.8%, 637.5%]
---------------------------------------------------------------------------------------------------------------------------------
Proportional difference in means, unadjusted (theta_N).
Intervals transform the endpoints of the interval of the log ratio. Covariance: robust to heteroskedasticity, convention ssc='stata'.
Rows: the estimation sample of the fit (11 of 12).
```

This table gives the unadjusted difference between observed promotion groups
within the price model's sample. The regression below conditions on the
included covariate. [Baseline balance example](../reference/results.md#inspect-balance-on-the-baseline-sample)

## Compare the estimates

```python
models = pm.etable(
    with_price, with_stock, names=("Adjust for price", "Adjust for stock"),
    stats=("nobs", "zeros"), stars=False, digits=4,
)
print(models)
```

```text
------------------------------------------------------
                    Adjust for price  Adjust for stock
------------------------------------------------------
Intercept                3.6730           -0.7143
                        (0.9998)          (0.8219)
promotion                0.5832            0.4017
                        (0.4652)          (0.2159)
price                   -0.3496
                        (0.1109)
stock                                      0.0783
                                          (0.0281)
------------------------------------------------------
Estimator                 PPML              PPML
Mean order (omega)         1                 1
Covariance               robust            robust
Convention               stata             stata
Reference                normal            normal
------------------------------------------------------
Observations               11                11
Zeros                    18.2%              9.1%
------------------------------------------------------
Standard errors in parentheses.
```

The promotion coefficients here can differ because the models use different
controls and different rows. To compare specifications on common eligible
rows, fit both models on that common input and check their retained positions
again. [Model table options](../reference/tables.md#tabulate-fitted-models)

## Export a table to Markdown

Copy the text from this call into a Markdown document to render the
first summary table as shown below.

```python
print(price_sample.to_markdown())
```

**Summary statistics. N = 11 rows**

|  | N | missing | mean | sd | p50 | zeros | mean \| \>0 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| visits | 11 | 0 | 5.00 | 3.92 | 5.00 | 18.2% | 6.11 |

*Notes:* Rows: the estimation sample of the fit \(11 of 12\).

[Explore results, effects and exports](reuse-a-fit.md)

<!-- guide-navigation:start -->
[Previous: Prepare data](../help/prepare-data.md) · [Guide index](../README.md) · [Next: Proportional changes](../methods/proportional-effects.md)
<!-- guide-navigation:end -->
