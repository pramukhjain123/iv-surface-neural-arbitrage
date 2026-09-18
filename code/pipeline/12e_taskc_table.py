"""
12e_taskc_table.py -- emit the Task C LaTeX table body from
results/phase6_taskC_summary.csv, so the manuscript table can be regenerated
from the data rather than retyped whenever another fold lands.

Writes paper/sections/taskc_table.tex, which results.tex \input{}s.
"""
from pathlib import Path
import pandas as pd
import importlib.util

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("fengler", _HERE / "11a_fengler.py")
FG = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(FG)
ROOT = FG.ROOT

ROWS = ["NN v3, band withheld", "NN v3, band in training", "Fengler + PCHIP",
        "SSVI + PCHIP", "SVI + PCHIP", "ThinPlate (native 2-D)",
        "CubicSpline + PCHIP"]
TEX = {"NN v3, band withheld": r"NN v3, band withheld",
       "NN v3, band in training": r"\quad{}\emph{same net, band in training}",
       "Fengler + PCHIP": r"Fengler $+$ PCHIP",
       "SSVI + PCHIP": r"SSVI $+$ PCHIP",
       "SVI + PCHIP": r"SVI $+$ PCHIP",
       "ThinPlate (native 2-D)": r"ThinPlate (native 2-D)",
       "CubicSpline + PCHIP": r"CubicSpline $+$ PCHIP"}


def main():
    df = pd.read_csv(ROOT / "results" / "phase6_taskC_summary.csv")
    c1 = df[df["window"] == "c1"].set_index("method")
    c2 = df[df["window"] == "c2"].set_index("method")
    n_folds = int(df["n_folds"].max())

    lines = []
    for m in ROWS:
        if m not in c1.index and m not in c2.index:
            continue
        def cell(t, col):
            if m not in t.index:
                return "--"
            v = float(t.loc[m, col])
            return f"{v:.2f}" if v < 100 else f"{v:.0f}"
        lines.append(f"{TEX[m]} & {cell(c1,'rmse_volpts')} & {cell(c1,'mae_volpts')} & "
                     f"{cell(c2,'rmse_volpts')} & {cell(c2,'mae_volpts')} \\\\")
        if m == "NN v3, band in training":
            lines.append(r"\midrule")

    body = "\n".join(lines)
    # Emit the complete float rather than just the row bodies: \input inside a
    # tabular opens a cell before booktabs' \midrule is read, which makes TeX
    # report "Misplaced \\noalign". Inputting a whole float at top level is safe.
    caption = (r"Task C, reconstructing the withheld $[30,60]$-day maturity band. "
               r"RMSE and MAE in volatility points, mean across " + str(n_folds) +
               r" folds (fold 16 excluded). c1 scores band slices dated inside the "
               r"training window, where all methods have matched same-day "
               r"information; c2 scores the test window, where the static network "
               r"must also extrapolate in date.")
    tex = (
        "\\begin{table}[!t]\n"
        "\\caption{" + caption + "}\n"
        "\\label{tab:taskc}\n\\centering\n\\footnotesize\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        "& \\multicolumn{2}{c}{c1 train window} & \\multicolumn{2}{c}{c2 test window} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
        "Method & RMSE & MAE & RMSE & MAE \\\\\n\\midrule\n"
        + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    out = Path(ROOT) / "paper" / "sections" / "taskc_table.tex"
    out.write_text(tex, encoding="utf-8")
    print(f"wrote {out}  ({n_folds} folds)")
    print(body)


if __name__ == "__main__":
    main()
