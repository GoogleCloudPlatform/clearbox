# Clearbox

ClearBox is a ranking tuning library with the primary use case of VAIS ranking
customization.

## How to use

```python
import pandas as pd
from clearbox import Trainer, Visualizer, RecallAtK, GridSearchLinearModel, \
  features as F
from clearbox.features import signals as S

# Read the data
qs_df = pd.read_csv('...')

# Create trainer and plotter utility objects.
trainer = Trainer(
    df=qs_df,
    seeds=[7, 15, 21, 42, 81],
    n_folds=3,
    metrics=[RecallAtK(k=1), RecallAtK(k=3), RecallAtK(k=5),],
    target_col='is_match',
    query_col='query',
)
visualizer = Visualizer(metric_list=trainer.metric_list)

# Compute baseline predictions
baseline_list = [
    (
        'base_rank',
        trainer.get_feature_baseline(feature=F.RR(-S.position, 40))
    ),
    (
        'gecko_score',
        trainer.get_feature_baseline(feature=S.gecko_score)
    ),
    (
        'jetstream_score',
        trainer.get_feature_baseline(feature=S.jetstream_score)
    ),
]

# Train grid search linear model and compare the results against the baseline.
visualizer.visualize_training_results(
    trainer.train(
        GridSearchLinearModel(metric=RecallAtK(3), n_opt_steps=10),
        features=[
            F.RR(S.gecko_score, 40.0),
            F.RR(S.jetstream_score, 40.0),
            F.RR(-S.position, 40.0),
        ],
    ),
    baseline_list,
)
```

## Key concepts

### Features

All the feature transformations should be done using `clearbox.features` module.
This way we can guarantee the tuned formula will be properly applied when
uploaded to the VAIS API server.

The feature transformation utilities of Clearbox rely on the `Node` classes and
the composition of those.

The following Node types are available:

-   `Constant(value: float)` -- evaluates to a constant vector of `value`.
-   `Signal(name: str, encode_values: bool)` -- a node that evaluates to a
    `name` column of the input dataframe, an input feature of the model to
    train. `encode_values` argument can be used to replace signal values with
    their unique integer IDs on the fly. This can be useful in rare cases (see
    `group_by` argument of `RR`), but should not be used for standard input
    features as such signals cannot be serialized to the ranking expression
    format.
-   `Log(arg: Node)` -- logarithmic transformation.
-   `Exp(arg: Node)` -- exponential transformation.
-   `RR(arg: Node, k: float, group_by: Node)` -- reciprocal rank transformation.
    First `arg` node is evaluated, then the result of the evaluation is sorted
    in non-ascending order and the ranks are produced. The final value for the
    i-th document is computed as `1 / (r_i + k)` where `r_i` is the rank of the
    i-th document in the sorted list.
-   `IsNaN(arg: Node)` -- converts the result of `arg` node evaluation to a
    vector containing 0 or 1 values. The i-th component is 1 if the i-th
    component of `arg` evaluation result is NaN, otherwise it's 0.
-   `FillNaN(arg: Node, fill_with: Node)` -- first we evaluate both arguments,
    then if i-th component of `arg` evaluation result is NaN, we replace it with
    i-th component of `fill_with` evaluation result.
-   `Add(args: list[Node])` -- sum up all evaluation results of all the nodes
    from `args` list.
-   `Mul(args: list[Node])` -- multiply all evaluation results of all the nodes
    from `args` list.

As standard Node API can be a bit verbose, we support the following shortcuts:

-   `node_1 + ... + node_n` is an equivalent of `Add([node_1, ..., node_n])`
-   `node_1 * ... * node_n` is an equivalent of `Mul([node_1, ..., node_n])`
-   `node * float` and `float * node` are compiled to `Mul([node,
    Constant(float)])`
-   `-node` is compiled to `Mul([node, Constant(-1.0)])`

Every `Node` subclass implements `serialize_to_ranking_expression` method which
compiles the subtree represented by the node into a ranking expression string.
The string can then be used to deploy the model, just pass it into the
`ranking_expression` field of the search request.

### Models

Currently only linear models are supported i.e. a trained model is always a
weighted sum of input features.

As all the models can be represented by a weighted sum of features, the
difference between the models is in a way those weights are trained. Currently
we support the following models:

-   `LinRegModel` -- linear regression, the weights are optimized using MSE
    minimization. Works reasonably well when there is a contiguous target (e.g.
    relevance score) and the metric you are targeting is contiguous by predicted
    score (e.g. nDCG). Arguments:
    -   `positive: bool` -- when set to `True`, forces the coefficients to be
        positive. Optional, `True` by default.
    -   `fit_intercept: bool` -- Whether to calculate the intercept for this
        model, disabled by default. Optional, `False` by default.
    -   `apply_scaler: bool` -- Whether to do the normalization of the features
        using `sklearn.preprocessing.StandardScaler`Optional, `True` by default.
-   `GridSearchLinearModel` -- the weights are optimized by an exhaustive search
    over a predefined grid of values. Works well when the metric you are trying
    to optimize isn't contiguous. Arguments:
    -   `metric: cbx_metrics.BaseMetric` -- The metric to optimize for.
    -   `grid_size: int` -- The number of steps to use for single feature in the
        grid. Optional, 10 by default.
    -   `print_progress: bool` -- Whether to print the progress of the grid
        search. Optional, `False` by default.
-   `BayesOptLinearModel` -- The weights are tuned using bayesian optimization
    over a predefined grid of values. We recommend to use it when the dataset is
    too large for Grid Search. Arguments:
    -   `surrogate_model: SurrogateModel` -- The surrogate model builder object
        to use for mean and standard deviation predictions. Options are:
        -   `GaussianProcessModel`
        -   `BayesianRidgeModel`
    -   `acquisition_function: AcquisitionFunction` -- Acquisition function
        object which is used to pick the next candidate points to compute the
        optimized function for. Options are:
        -   `ExpectedImprovement`
        -   `MaxMean`
    -   `metric` -- The metric to optimize, the values of the metric are used as
        targets in the optimization process.
    -   `n_opt_steps` -- Number of the optimization steps to perform.
    -   `batch_size` -- Number of candidates pick on each optimization step.
    -   `seed_batch_size` -- Number of the candidates to compute the `metric`
        value for before the first optimization step. Those values are then used
        as the initial training set for the `surrogate_model`.
    -   `grid_size` -- The number of steps to use for single feature in the
        grid. In the canonical implementation of BO the candidates are sampled
        from the multivariate normal distribution at each optimization step.
        But, as our implementation is basically a logical evolution of Grid
        Search algorithm, we retrain the same uniform grid and use it as a
        source of candidates.
    -   `print_progress` -- Whether to print the progress of the optimization.

### Metrics

The following metrics are available out of the box:

-   `RecallAtK(k: int, bin_threshold: float = 0.5)` -- **Recall@K**, `k`
    argument is mandatory, `bin_threshold` can be used to control floating point
    to {0, 1} label conversion.
-   `NDCG(k: int | None = None, pw_decay_factor: float = 1.0)` -- **NDCG** if
    `k` is not provided, otherwise NDCG@K. `pw_decay_factor` argument can be
    used to apply multiplicative position-weighted decay to the relevance
    target.
-   `AUROC()` -- **area under ROC**.

Each metric class inherits from `BaseMetric` and overrides 2 abstract methods:

-   `name` property -- it should return the name of the metric as `str`.
-   `compute` method -- it accepts the following arguments:

    -   `query_ids: NDArray[int]` -- integer query IDs of query & doc pairs.
    -   `scores: NDArray[float]` -- predicted scores for all query & doc pairs.
    -   `targets: NDArray[float]` -- ground truth scores for all query & doc
        pairs.

    The method should return a single `float` metric value.

To create a custom metric, create a new class that inherits from `BaseMetric`
and implement `name` and `compute`.

The intended usage of the library does not imply invoking neither `name` nor
`compute` directly as it should be done internally by the `Trainer` abstraction.

### Trainer

`Trainer` class is the main abstraction over the cross-validation and training
loop. It ties together all the pieces we've described before and uses them to
train the model. Its `__init__` method supports the following arguments:

-   `df: pd.DataFrame` -- Whole dataset as a data frame.
-   `seeds: list[int]` -- List of random seed integers. For each of the seeds we
    generate a separate validation split of `n_folds` and train `n_folds`
    models.
-   `n_folds: int` -- Number of folds in one cross validation split.
-   `metrics: list[BaseMetric]` -- List of metrics to compute for each training
    iteration.
-   `target_col: str` -- Column to use as an optimization target.
-   `query_col: str` -- Column to use as a query ID.
-   `rank_col: str` -- Column representing the golden ranking, optional and
    defaults to `target_col`. It allows defining optimization target different
    from the golden ranking column.
-   `stratify_by_col: str` -- Column to use for stratification, i.e. the we'll
    do our best to balance the distribution of this column across the folds.
-   `test_df: pd.DataFrame` -- Optional dataset to run final model evaluation
    on.

`train` method is used to train the ranking model from scratch, the following
arguments are available:

-   `model_builder: cbx_models.Model` -- model to train, see "Models" section
    for more details.
-   `features: list[F.Node]` -- list of instances of nodes to use as features,
    see "Features" section for more details.
-   `df: pd.DataFrame` -- optional, overrides the corresponding argument of
    `__init__`.
-   `metrics: list[cbx_metrics.BaseMetric]` -- optional, overrides the
    corresponding argument of `__init__`.
-   `target_col: str` -- optional, overrides the corresponding argument of
    `__init__`.
-   `query_col: str` -- optional, overrides the corresponding argument of
    `__init__`.
-   `rank_col: str` -- optional, overrides the corresponding argument of
    `__init__`.
-   `stratify_by_col: str` -- optional, overrides the corresponding argument of
    `__init__`.
-   `test_df: pd.DataFrame` -- optional, dataframe to use for model evaluation
    after the training is done.
-   `print_progress: bool` -- if `True`, print the progress of the training to
    stdout.
-   `num_parallel_workers: int | t.Literal['auto']` -- optional, number of
    workers to use for training parallelization. Should be > 1, 1 means no
    parallelization. String `"auto"` can be passed to pick the number of workers
    automatically based on the number of CPU cores.
