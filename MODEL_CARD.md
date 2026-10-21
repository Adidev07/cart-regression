# Model / system card

## Intended use

Train a depth-limited regression tree on nonlinear synthetic demand and inspect the learned rules. Educational local experiment; no production deployment is implied.

## Data

Synthetic arrays, manually written documents or replayed JSON. No real user records. Seeds and sample inputs are recorded in app.py and sample_input.json.

## Method

At every node search feature thresholds at empirical quantiles and minimize total squared error. Stop at maximum depth, small node size or insufficient gain. Leaf predictions are training means.

## Evaluation protocol

Choose depth 2, 4 or 6 using a separate validation split; report held-out RMSE against a training-mean baseline. Inputs and targets are generated from a nonlinear process with noise.

## Known limits

Quantile candidates approximate exhaustive CART. Small synthetic data cannot establish production forecasting performance; random splits assume independent rows.

## Verification status

Local tests and default experiment are executed during packaging. Live model generation is not exercised in the build. A successful validation result is limited to the checks implemented in code.

## Further work

Replace the generator with licensed tabular data; compare a linear model and random forest, add missing-value handling, and evaluate subgroup errors.
