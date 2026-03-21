from pathlib import Path
import pandas as pd
import re

'''
Input papers: 57157
Filtered papers: 837
Saved to: save/papers_all_keyword_filtered.csv
'''


#  Keyword patterns 

llm_pattern = re.compile(
    r"\bgenai\b|\blanguage\s+model(s)?\b|\bllm(s)?\b",
    flags=re.IGNORECASE
)

study_pattern = re.compile(
    r"\bstudy\b",
    flags=re.IGNORECASE
)

user_pattern = re.compile(
    r"\b(user|participants?|humans?)\b",
    flags=re.IGNORECASE
)


input_file = Path("data") / "papers_all.csv"
output_file = Path("save") / "papers_all_keyword_filtered.csv"


def filter_dataframe(df):
    # Keep only years 2022-2025
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df[df["Year"].between(2022, 2025)]

    # Fill missing text columns
    df["Title"] = df["Title"].fillna("")
    df["Abstract"] = df["Abstract"].fillna("")


    llm_mask = (
        df["Title"].str.contains(llm_pattern, regex=True) |
        df["Abstract"].str.contains(llm_pattern, regex=True)
    )

    study_mask = (
        df["Title"].str.contains(study_pattern, regex=True) |
        df["Abstract"].str.contains(study_pattern, regex=True)
    )

    user_mask = (
        df["Title"].str.contains(user_pattern, regex=True) |
        df["Abstract"].str.contains(user_pattern, regex=True)
    )


    df = df[llm_mask & study_mask & user_mask]
    return df[["Venue", "Year", "Title", "Authors", "Abstract", "URL"]]


def main():
    if not input_file.exists():
        print(f"File not found: {input_file}")
        return

    df = pd.read_csv(input_file)
    filtered_df = filter_dataframe(df)

    filtered_df = filtered_df.drop_duplicates(subset=["URL"])

    filtered_df.to_csv(output_file, index=False)

    print(f"Input papers: {len(df)}")
    print(f"Filtered papers: {len(filtered_df)}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()