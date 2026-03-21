import pandas as pd
import re
from datetime import datetime
import os

# ========= Keyword =========


llm_pattern = re.compile(
    r"\bgenai\b|\blanguage\s+models?\b|\bllms?\b",
    flags=re.IGNORECASE
)

study_pattern = re.compile(
    r"\bstudy\b",
    flags=re.IGNORECASE
)

user_pattern = re.compile(
    r"\b(?:user|participants?|humans?)\b",
    flags=re.IGNORECASE
)

MASTER_FILE = "/scratch/nicole/human_subject/output/master_keyword_filtered.csv"


def filter_dataframe(df):
    current_year = datetime.now().year

   
    required_columns = ["Venue", "Year", "Title", "Abstract", "URL"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


    df = df.copy()

    # Year filtering
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df[df["Year"].between(2022, current_year)].copy()

    df["Title"] = df["Title"].fillna("").astype(str)
    df["Abstract"] = df["Abstract"].fillna("").astype(str)


    llm_mask = (
        df["Title"].str.contains(llm_pattern, na=False) |
        df["Abstract"].str.contains(llm_pattern, na=False)
    )

    study_mask = (
        df["Title"].str.contains(study_pattern, na=False) |
        df["Abstract"].str.contains(study_pattern, na=False)
    )

    user_mask = (
        df["Title"].str.contains(user_pattern, na=False) |
        df["Abstract"].str.contains(user_pattern, na=False)
    )

    df = df[llm_mask & study_mask & user_mask].copy()

    return df[["Venue", "Year", "Title", "Abstract", "URL"]]


def main():
    input_path = input("Enter path to input CSV file: ").strip()

    if not os.path.exists(input_path):
        print("File not found.")
        return

    sep_answer = input("Is it tab-separated? (y/n): ").strip().lower()
    sep = "\t" if sep_answer == "y" else ","

    try:
        df = pd.read_csv(input_path, sep=sep)
        filtered_df = filter_dataframe(df)
    except Exception as e:
        print(f"Error: {e}")
        return

    print(f"Filtered rows from this file: {len(filtered_df)}")


    if os.path.exists(MASTER_FILE):
        try:
            master_df = pd.read_csv(MASTER_FILE)
            combined = pd.concat([master_df, filtered_df], ignore_index=True)
        except Exception as e:
            print(f"Error reading master file: {e}")
            return
    else:
        combined = filtered_df.copy()

    # Deduplicate by URL
    combined["URL"] = combined["URL"].fillna("").astype(str)
    combined = combined.drop_duplicates(subset=["URL"])

    os.makedirs(os.path.dirname(MASTER_FILE), exist_ok=True)

    combined.to_csv(MASTER_FILE, index=False)

    print(f"Total rows in master file after merge: {len(combined)}")
    print(f"Saved to {MASTER_FILE}")


if __name__ == "__main__":
    main()
    