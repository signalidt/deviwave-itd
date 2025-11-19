#!/usr/bin/env python3
"""
step4_hourly_stat.py

Generate 24-hour segmented behavior statistics for each user.
"""

import os
import argparse
import logging
import pandas as pd
import yaml


# -------------------------------
# Load YAML config
# -------------------------------
def load_config(path: str) -> dict:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Failed to load config file {path}: {e}")
        raise


# -------------------------------
# Hourly statistics generator
# -------------------------------
def process_hourly_stats(config: dict):

    source_folder = config['source_folder']
    save_folder = config['save_folder']
    behavior_types = config.get('behavior_types', [])

    os.makedirs(save_folder, exist_ok=True)
    logging.info(f"Save folder ensured: {save_folder}")

    # -------------------------------
    # Iterate scenarios
    # -------------------------------
    for scenario in os.listdir(source_folder):
        scenario_path = os.path.join(source_folder, scenario)
        if not os.path.isdir(scenario_path):
            continue

        logging.info(f"Processing scenario: {scenario}")

        # -------------------------------
        # Iterate user folders
        # -------------------------------
        for user_folder in os.listdir(scenario_path):
            user_path = os.path.join(scenario_path, user_folder)
            if not os.path.isdir(user_path):
                continue

            logging.info(f"  Processing user: {user_folder}")

            updated_behavior_file = os.path.join(user_path, "behavior_with_label.csv")
            if not os.path.isfile(updated_behavior_file):
                logging.warning(f"    No updated_behavior.csv in {user_path}")
                continue

            # -------------------------------
            # Step 1: Load & preprocess
            # -------------------------------
            try:
                df = pd.read_csv(updated_behavior_file)
            except Exception as e:
                logging.error(f"Failed to read {updated_behavior_file}: {e}")
                continue

            # Required columns check
            if not {'date', 'behavior', 'label'}.issubset(df.columns):
                logging.error(f"    Missing required columns in {updated_behavior_file}")
                continue

            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['date_only'] = df['date'].dt.date
            df['hour'] = df['date'].dt.hour

            # -------------------------------
            # Step 2: Group hourly × behavior
            # -------------------------------
            hourly = df.groupby(['date_only', 'hour', 'behavior']).size().unstack(fill_value=0)

            # -------------------------------
            # Step 3: Ensure all behaviors exist
            # -------------------------------
            for b in behavior_types:
                if b not in hourly.columns:
                    hourly[b] = 0

            hourly = hourly.astype(int).reset_index()

            # Rename for clarity
            rename_map = {b: f"{b}_count" for b in behavior_types}
            hourly = hourly.rename(columns=rename_map)

            # -------------------------------
            # Step 4: Ensure full 0–23 hours for each date
            # -------------------------------
            dates = df['date_only'].unique()
            full_index = pd.MultiIndex.from_product([dates, range(24)],
                                                    names=['date_only', 'hour'])

            hourly = hourly.set_index(['date_only', 'hour']) \
                           .reindex(full_index, fill_value=0) \
                           .reset_index()

            # -------------------------------
            # Step 5: Add total behavior count
            # -------------------------------
            hourly['total_behavior_count'] = hourly[[f"{b}_count" for b in behavior_types]].sum(axis=1)

            # -------------------------------
            # Step 6: Daily label
            # -------------------------------
            label_info = df.groupby('date_only')['label'].max().reset_index()
            label_info['label'] = (label_info['label'] > 0).astype(int)

            hourly = hourly.merge(label_info, on='date_only', how='left')

            # -------------------------------
            # Step 7: Sort & save
            # -------------------------------
            hourly = hourly.sort_values(['date_only', 'hour'])

            output_file = os.path.join(save_folder, f"{scenario}_{user_folder}_hourly.csv")
            try:
                hourly.to_csv(output_file, index=False)
                logging.info(f"    Saved: {output_file}")
            except Exception as e:
                logging.error(f"Failed to save {output_file}: {e}")


# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate hourly behavior statistics")
    parser.add_argument('-c', '--config', required=True, help="YAML config file path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s')

    full_conf = load_config(args.config)
    config = full_conf.get('step4_hourly_stat', full_conf)

    process_hourly_stats(config)


if __name__ == "__main__":
    main()
