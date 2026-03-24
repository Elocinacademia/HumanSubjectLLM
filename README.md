# HumanSubjectLLM

## Repository Structure

```text
HumanSubjectLLM/
├── crawlers/                     # Crawlers or scripts for collecting source data
├── save/                         # Saved outputs, intermediate files, or processed results
├── cleaning.py                   # Data cleaning and preprocessing
├── download_all_pdfs.py          # Bulk PDF download utility
├── pip1_keyword_filtering.py     # Pipeline 1 Step 1 keyword-based filtering
├── pip1_abstract_judge.py        # Pipeline 1 Step 2 abstract-level LLM judging
├── pip1_fullpaper_judge.py       # Pipeline 1 Step 3 full-paper LLM judging
├── pip2_keyword_filtering.py     # Pipeline 2 Step 1 keyword-based filtering
├── pip2_fullpaper_judge.py       # Pipeline 2 Step 2 full-paper LLM judging
├── LICENSE
└── README.md
```


## Data

```text
Put CSV files into local dir, e.g. data\
```

## Pipeline 2 STEPS
**Step 1**:

Run 
```
python cleaning.py
```
to merge all CSV files into a single file. This step also removes unwanted ML conferences and keeps only the selected ones.

**Output**: 
```
save/papers_all.csv
```

| File | Rows Kept |
|---|---|
| `soups_2022_2025.csv` | 133 |
| `aaai_2022.csv` | 1624 |
| `all_ml_nlp_papers.csv` | 32814 |
| `chi-uist-iui.csv` | 4675 |
| `ijcai_papers.csv` | 4038 |
| `aaai_2023_2025.csv` | 7415 |
| `all_security_papers.csv` | 5891 |
| `popets_2022_2025.csv` | 567 |
| Total | 57157 |

**Step 2**: 

Run 
```
pip2_keyword_filtering.py
```
to filter papers published between 2022 and 2025. A paper is kept only if its title or abstract matches all three keyword groups: (1) LLM-related terms (`genai`, `language model`, `llm`), (2) `study`, and (3) human-related terms (`user`, `participant`, `human`).

**Output:** 
```
save/papers_all_keyword_filtered.csv
```
-> 837 papers in total

**Step 3**: 

Run 
```
pip2_fullpaper_judge.py
```
to evaluate the full papers. This script uses gpt-4o through the OpenAI API, so please set your API key in the environment before running it.

> [!IMPORTANT]
> Before running this step, first execute `download_all_pdfs.py` to download the PDFs for the papers. This script will automatically create a `/pdfs` directory.
>
> You also need to replace `COOKIES_FILE` with your own cookies file, as access to ACM Digital Library requires valid login credentials.

**Output:** `judged_fullpaper.csv`
