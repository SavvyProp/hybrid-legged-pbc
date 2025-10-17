from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def _infer_xy(df: pd.DataFrame, x_col: Optional[str], y_col: Optional[str]) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """Infer x and y series and their column names from a DataFrame.

    Preference:
    - x: 'step' > 'time' > 't' > index
    - y: provided y_col, else 'alive' if present, else last numeric col not x.
    """
    cols = df.columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Determine x column
    if x_col and x_col in cols:
        x_name = x_col
    elif 'step' in cols:
        x_name = 'step'
    elif 'time' in cols:
        x_name = 'time'
    elif 't' in cols:
        x_name = 't'
    else:
        x_name = None  # use index

    # Determine y column
    if y_col and y_col in cols:
        y_name = y_col
    elif 'alive' in cols:
        y_name = 'alive'
    else:
        # use last numeric column that isn't x
        candidates = [c for c in num_cols if c != x_name]
        if not candidates and num_cols:
            candidates = num_cols
        y_name = candidates[-1] if candidates else cols[-1]

    # Extract arrays
    x = df[x_name].to_numpy() if x_name else np.arange(len(df))
    y = df[y_name].to_numpy()
    return x, y, (x_name or 'index'), y_name


def plot_ft_pd_alive(
    ft_csv: Optional[Path] = None,
    pd_csv: Optional[Path] = None,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    title: str = 'FT vs PD',
    save_path: Optional[Path] = None,
    show: bool = True,
):
    """Plot both data/eval/ft_constant_frc_alive.csv and data/eval/pd_constant_frc_alive.csv on the same axes.

    Args:
      ft_csv: Path to FT CSV. Defaults to ./data/eval/ft_constant_frc_alive.csv
      pd_csv: Path to PD CSV. Defaults to ./data/eval/pd_constant_frc_alive.csv
      x_col: Optional x column name to use.
      y_col: Optional y column name to use.
      title: Plot title.
      save_path: If provided, saves the figure to this path.
      show: If True, calls plt.show().

    Returns:
      (fig, ax)
    """
    here = Path(__file__).parent
    ft_csv = Path(ft_csv) if ft_csv else here / 'data' / 'eval' / 'ft_constant_frc_alive.csv'
    pd_csv = Path(pd_csv) if pd_csv else here / 'data' / 'eval' / 'pd_constant_frc_alive.csv'

    df_ft = pd.read_csv(ft_csv)
    df_pd = pd.read_csv(pd_csv)

    x_ft, y_ft, x_name_ft, y_name_ft = _infer_xy(df_ft, x_col, y_col)
    x_pd, y_pd, x_name_pd, y_name_pd = _infer_xy(df_pd, x_col, y_col)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x_ft, y_ft, label=f'FT ({y_name_ft})', linewidth=2)
    ax.plot(x_pd, y_pd, label=f'PD ({y_name_pd})', linewidth=2)

    # Axis labels
    ax.set_xlabel(x_col or (x_name_ft if x_name_ft == x_name_pd else 'x'))
    ax.set_ylabel(y_col or (y_name_ft if y_name_ft == y_name_pd else 'value'))
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)

    if show:
        plt.show()

    return fig, ax


if __name__ == '__main__':
    plot_ft_pd_alive()