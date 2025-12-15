import sys
from pathlib import Path

# Add project root so src can be imported
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import sys
import pandas as pd
from pathlib import Path
from src.ml.predict import predict_from_values, get_feature_names

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python tools/predict_cli.py 10,20,30,...")
        print("  python tools/predict_cli.py input.csv")
        return

    argument = sys.argv[1]

    # CASE 1: CSV input
    if argument.endswith(".csv") and Path(argument).exists():
        df = pd.read_csv(argument)
        row = df.select_dtypes(include=["number"]).iloc[0].values.tolist()
        result = predict_from_values(row)
        print("Prediction from CSV:", result)
        return

    # CASE 2: Comma-separated values
    try:
        values = [float(x) for x in argument.split(",")]
        feat_count = len(get_feature_names())

        if len(values) != feat_count:
            print(f"Error: expected {feat_count} numeric values, got {len(values)}")
            return

        result = predict_from_values(values)
        print("Prediction:", result)

    except Exception as e:
        print("Error parsing input:", e)

if __name__ == "__main__":
    main()

