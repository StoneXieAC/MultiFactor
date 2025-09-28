import pandas as pd
import warnings
import backtest

warnings.filterwarnings("ignore")


def run(data_path, output_path):
    df = pd.read_parquet(data_path)

    factor_list = [col for col in df.columns if col != 'label']
    results = []

    for fac in factor_list:
        print(f"\n===== Backtesting factor: {fac} =====")

        df_fac = df[[fac, 'label']].copy()
        df_fac.columns = ['factor', 'label']

        (
            ic_mean,
            ic_std,
            ic_ir,
            ann_l,
            ann_ls,
            sharpe_l,
            sharpe_ls
        ) = backtest.run(df_fac)

        print(f"IC Mean: {ic_mean:.4f}, IC Std: {ic_std:.4f}, IC IR: {ic_ir:.4f}")
        print(f"Annualized Long Return: {ann_l:.2%}, Annualized Long-Short Return: {ann_ls:.2%}")
        print(f"Long Sharpe Ratio: {sharpe_l:.4f}, Long-Short Sharpe Ratio: {sharpe_ls:.4f}")

        results.append({
            "factor": fac,
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "ic_ir": ic_ir,
            "ann_long": ann_l,
            "ann_ls": ann_ls,
            "sharpe_long": sharpe_l,
            "sharpe_ls": sharpe_ls
        })

    results = pd.DataFrame(results)
    results.to_csv(output_path, index=False)

    print("\nAll factor backtests completed! Results saved to:", output_path)


if __name__ == "__main__":
    data_path = './data/factor_train.parquet'
    output_path = 'data/single_result.csv'
    run(data_path, output_path)
