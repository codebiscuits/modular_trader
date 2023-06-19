import pandas as pd
from pathlib import Path
from pprint import pprint
import statistics as stats
import plotly.express as px

pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.precision', 4)
pd.options.plotting.backend = "plotly"

res_path = Path("/machine_learning/rfc_results/")

def choose_best(df, col):
    df = (
        df.sort_values(col, ascending=False)
        .loc[df.trades_taken > 30]
        .reset_index(drop=True)
        .head(int(len(df) // 2))
    )
    print(df)

    if len(df) > 3:
        # print(df_precision)
        fw_dict = dict(
            top_frac_width=float(df.frac_width.iat[0]),
            mean_frac_width=df.frac_width.mean(),
            med_frac_width=df.frac_width.median(),
            mode_frac_width=float(df.frac_width.mode().iat[0]),
        )

        sp_dict = dict(
            top_spacing=float(df.spacing.iat[0]),
            mean_spacing=df.spacing.mean(),
            med_spacing=df.spacing.median(),
            mode_spacing=float(df.spacing.mode().iat[0]),
        )

        print(f"{col} best frac_width: {round_odd(stats.mean(fw_dict.values()))}, "
              f"best spacing: {round(stats.mean(sp_dict.values()))}")


def round_odd(x):
    return (round((x-1) / 2) * 2) + 1


def pnl_corr_best(df):
    correl_dict = dict(
    pos_pred = df.pnl.corr(df.pos_preds),
    prec = df.pnl.corr(df.precision),
    recall = df.pnl.corr(df.recall),
    f1 = df.pnl.corr(df.f1),
    f_beta = df.pnl.corr(df.f_beta),
    auroc = df.pnl.corr(df.auroc),
    ppp = df.pnl.corr(df.pos_preds*df.precision),
    ppfb = df.pnl.corr(df.pos_preds*df.f_beta),
    )

    # for i in ['pos_preds', 'precision', 'recall', 'f1', 'f_beta', 'auroc']:
    #     fig = df.plot.scatter(x='pnl', y=i)
    #     fig.show()

    return correl_dict

all_corr = {}
for p in res_path.glob('*'):
    print('\n\n', p.stem)
    # if p.stem.split('_')[0] == 'short':
    df = pd.read_parquet(p)
    # print(df.columns)
    choose_best(df, 'precision')
    choose_best(df, 'f_beta')
    choose_best(df, 'win_rate')
    choose_best(df, 'pnl')
    # all_corr[p.stem] = pnl_corr_best(df)

all_corr_df = pd.DataFrame(all_corr)
print(all_corr_df)
print(all_corr_df.mean(axis=1))

