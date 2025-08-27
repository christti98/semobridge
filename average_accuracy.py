import pandas as pd
import argparse
import numpy as np


def parse_time_str(time_str):
    """Parse time format hh:mm:ss to seconds."""
    h, m, s = map(int, time_str.split(":"))
    return h * 3600 + m * 60 + s


def format_time(seconds):
    """Convert seconds to hh:mm:ss format."""
    return (
        f"{int(seconds//3600):02d}:{int((seconds%3600)//60):02d}:{int(seconds%60):02d}"
    )


def analyze_results(csv_path, target_shot):
    # Load CSV
    df = pd.read_csv(csv_path)

    # Filter rows by desired shot count
    df_shot = df[df["shots"] == target_shot]

    # Compute average accuracy
    avg_acc = df_shot["accuracy"].mean()

    # Compute average of reported stds
    avg_std = df_shot["accuracy_std"].mean()

    # Compute average train time
    train_times_sec = df_shot["train_time"].apply(parse_time_str)
    avg_time_sec = train_times_sec.mean()
    avg_time_str = format_time(avg_time_sec)

    # Print results
    print(f"Results for {target_shot}-shot setting:")
    print(f"  - Average Accuracy:         {avg_acc:.2f}%")
    print(f"  - Average Accuracy StdDev:  {avg_std:.2f}%")
    print(f"  - Average Training Time:    {avg_time_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to CSV file")
    parser.add_argument(
        "--shots", type=int, required=True, help="Target K-shot value (e.g., 1, 4, 16)"
    )
    args = parser.parse_args()

    analyze_results(args.csv_path, args.shots)
