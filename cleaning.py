from pathlib import Path
import pandas as pd

'''
Processing: soups_2022_2025.csv
  Final rows kept: 133

Processing: aaai_2022.csv
  Final rows kept: 1624

Processing: all_ml_nlp_papers.csv
  Filtered all_ml_nlp_papers.csv: 213510 -> 32814 papers
  Final rows kept: 32814

Processing: chi-uist-iui.csv
  Final rows kept: 4675

Processing: ijcai_papers.csv
  Final rows kept: 4038

Processing: aaai_2023_2025.csv
  Final rows kept: 7415

Processing: all_security_papers.csv
  Final rows kept: 5891

Processing: popets_2022_2025.csv
  Final rows kept: 567

Total papers in combined file: 57157
'''




data_dir = Path("data")
output_file = data_dir / "papers_all.csv"

target_cols = ["Venue", "Year", "Title", "Authors", "Abstract", "URL"]

# For nlp papers
target_venues = {
    "Findings of the Association for Computational Linguistics: ACL 2023",
    "Findings of the Association for Computational Linguistics: ACL 2024",
    "Findings of the Association for Computational Linguistics: ACL 2025",
    "Findings of the Association for Computational Linguistics: EMNLP 2023",
    "Findings of the Association for Computational Linguistics: EMNLP 2024",
    "Findings of the Association for Computational Linguistics: NAACL 2024",
    "Findings of the Association for Computational Linguistics: NAACL 2025",
    "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
}

column_aliases = {
    "Venue": ["Venue", "venue"],
    "Year": ["Year", "year"],
    "Title": ["Title", "title"],
    "Authors": ["Authors", "Author", "authors", "author"],
    "Abstract": ["Abstract", "abstract", "Summary", "summary"],
    "URL": ["URL", "Url", "url", "Link", "link"],
}

dfs = []

for csv_file in data_dir.glob("*.csv"):
    if csv_file.name == "papers_all.csv":
        continue

    print(f"\nProcessing: {csv_file.name}")

    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()

    print("  Original columns:", list(df.columns))


    rename_map = {}
    for target_col, candidates in column_aliases.items():
        for candidate in candidates:
            if candidate in df.columns:
                rename_map[candidate] = target_col
                break

    print("  Matched columns:", rename_map)

    df = df[list(rename_map.keys())].rename(columns=rename_map)

    # Make sure all target columns exist
    for col in target_cols:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[target_cols]

    # NLP papers
    if csv_file.name == "all_ml_nlp_papers.csv":
        before_count = len(df)
        df["Venue"] = df["Venue"].astype(str).str.strip()
        df = df[df["Venue"].isin(target_venues)]
        print(f"  Filtered all_ml_nlp_papers.csv: {before_count} -> {len(df)} papers")

    print(f"  Final rows kept: {len(df)}")
    dfs.append(df)

papers_all = pd.concat(dfs, ignore_index=True)
papers_all.to_csv(output_file, index=False)

print(f"\nTotal papers in combined file: {len(papers_all)}")
print(f"Saved combined file to: {output_file}")