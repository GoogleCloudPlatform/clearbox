## Clear Box

ClearBox provides advanced customization capabilities for the ranking formula
within Vertex AI Search. Refine result relevance by adjusting weights and
incorporating new signals tailored to your specific dataset and user
requirements.

Take a look at the [docs](./docs/index.md) for more information.

### Installation

```shell
pip install "git+https://github.com/GoogleCloudPlatform/clearbox.git"
```

### How to use

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
