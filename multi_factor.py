import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
import backtest

warnings.filterwarnings("ignore")


def select_global_lgb(df, factors, top_n=5):
    """
    Select top_n factors using global LightGBM model
    Returns selected factors and their importance
    """
    X = df[factors]
    y = df['label']

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1,
        random_state=42,
    )
    model.fit(X, y)

    imp = pd.Series(model.feature_importances_, index=factors)
    imp = imp.sort_values(ascending=False)

    sel_fac = imp.head(top_n).index.tolist()
    print(f"LightGBM selected {len(sel_fac)}/{len(factors)} factors")
    print("Selected factors (sorted by importance):")
    print(sel_fac)

    return sel_fac, imp


def scores_global(df, factors):
    """Calculate scores using fixed factor pool"""
    Z = df[factors].apply(lambda x: (x - x.mean()) / x.std())
    df_scores = pd.DataFrame(index=df.index)
    df_scores["score"] = Z.mean(axis=1)
    return df_scores


def run(train_path, test_path, imp_path, res_path):
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna()
    test_df = test_df.replace([np.inf, -np.inf], np.nan).dropna()

    factors = [c for c in train_df.columns if c != 'label']

    selected_factors, importance = select_global_lgb(train_df, factors, top_n=5)

    scores_df = scores_global(test_df, selected_factors)

    scores_df = scores_df.rename(columns={'score': 'factor'}).join(test_df[['label']], how='inner')

    ic_mean, ic_std, ic_ir, ann_l, ann_ls, sharpe_l, sharpe_ls = backtest.run(scores_df)

    print("\n===== Backtest Results (LightGBM Selected Factors, Test Set 2024-2025) =====")
    print(f"IC Mean: {ic_mean:.4f}, IC Std: {ic_std:.4f}, IC IR: {ic_ir:.4f}")
    print(f"Annualized Long Return: {ann_l:.2%}")
    print(f"Annualized Long-Short Return: {ann_ls:.2%}")
    print(f"Long Sharpe Ratio: {sharpe_l:.4f}")
    print(f"Long-Short Sharpe Ratio: {sharpe_ls:.4f}")

    importance = importance.reset_index()
    importance.columns = ['factor', 'importance']
    importance.to_csv(imp_path, index=False)
    print(f"Factor importance saved to: {imp_path}")

    results = [{
        "factor": "multifactor",
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_ir": ic_ir,
        "ann_l": ann_l,
        "ann_ls": ann_ls,
        "sharpe_l": sharpe_l,
        "sharpe_ls": sharpe_ls
    }]

    results = pd.DataFrame(results)
    results.to_csv(res_path, index=False)

    print("Results saved to:", res_path)


if __name__ == "__main__":
    train_path = "./data/factor_train.parquet"
    test_path = "./data/factor_test.parquet"
    imp_path = "./data/factor_imp.csv"
    res_path = "./data/multi_result.csv"
    run(train_path, test_path, imp_path, res_path)
