# Intelligent Marketing — Competition Baseline

Notebook baseline for predicting content interaction counts using tabular, date, text, and author-history features with CatBoost regression.

## Setup and data

```bash
git clone https://github.com/Shayne-Pro/spring_camp_intelligent_marketing.git
cd spring_camp_intelligent_marketing
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyter lab
```

Open `智慧营销_baseline.ipynb`. Obtain the authorized competition data separately and place the extracted files at:

```text
智慧赢销dataset/
  train.txt
  A.txt
```

The notebook's optional unzip cell expects `智慧赢销dataset.zip` beside it. Skip extraction when the directory is already prepared. Data files are tab-separated.

Training columns: `id`, `interaction_cnt`, `uid`, `site_id`, `gender`, `age`, `city`, `post_type`, `fans_cnt`, `video_cnt`, `coin_cnt`, `publish_time`, `update_time`, `title`, `content`, `cover_ocr_content`, `video_content`. Test data requires the same input columns except the target. Dates are parsed as `YYYYMMDD`.

## Pipeline and outputs

1. Parse dates, counts, ratios, and content-update intervals.
2. Generate TF-IDF/SVD text features and cross-fold author statistics.
3. Fit five CatBoost folds and average test predictions.
4. Round, clip negative predictions, and apply the historical final scaling by `1/2`.
5. Write `final_results.txt` with `id` and `interaction_cnt`, separated by tabs.

The final scaling is inherited from this baseline, not a generally justified calibration. Review it against the current competition rules before submitting predictions.

## Evaluation limitations

The target-statistics folds are generated separately from the randomized model folds, and text transforms are fitted before model cross-validation. Consequently the printed OOF MAE is not a leakage-free estimate. A stricter experiment must fit target encoders and text transforms within the model's training folds and use a fixed random seed. No new leaderboard score is claimed by this maintenance update.

The count preprocessing preserves numeric counts and maps the special `小于100` bucket to `50`; unknown values are filled with zero. This fixes the previous mapping that silently converted numeric counts to zero.

No dataset, fitted model, or benchmark execution is provided. The repository currently has no repository-level license grant.

## Regression checks

```bash
python -m unittest discover -s tests -v
```
