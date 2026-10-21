import pandas as pd
from pathlib import Path

def save_df(df: pd.DataFrame, output_path: Path) -> None:
    """Saves a DataFrame to CSV or Excel based on file extension."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".csv":
        df.to_csv(output_path, index=False)
    elif output_path.suffix in [".xlsx", ".xls"]:
        df.to_excel(output_path, index=False)
    else:
        df.to_csv(output_path.with_suffix(".csv"), index=False)
