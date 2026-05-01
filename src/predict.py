from __future__ import annotations

import argparse
import json

from pipeline import predict_risk


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict aviation risk from input features.")
    parser.add_argument("--year", type=float, default=2020)
    parser.add_argument("--month", type=float, default=6)
    parser.add_argument("--aboard", type=float, default=120)
    parser.add_argument("--summary_length", type=float, default=100)
    parser.add_argument("--is_military", type=int, default=0)
    parser.add_argument("--is_commercial", type=int, default=1)
    parser.add_argument("--weather_mentioned", type=int, default=0)
    parser.add_argument("--mechanical_failure", type=int, default=0)
    parser.add_argument("--pilot_error", type=int, default=0)
    parser.add_argument("--aircraft_category", type=str, default="Commercial")
    parser.add_argument("--country", type=str, default="United States")
    parser.add_argument("--broad_phase", type=str, default="Unknown")
    args = parser.parse_args()

    payload = vars(args)
    result = predict_risk(payload)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
