import pandas as pd
import json
import os
import re
from openai import OpenAI


MODEL = "gpt-4o-mini-2024-07-18"

INPUT_FILE = "/scratch/nicole/human_subject/output/master_keyword_filtered.csv"   
OUTPUT_FILE = "/scratch/nicole/human_subject/output/papers_llm_judged.csv"             

SAVE_EVERY = 20  


# =========================
# LLM Judge
# =========================
client = OpenAI(api_key=API_KEY)

SYSTEM_MSG = (
    "You are a strict screening assistant for a systematic review.\n"
    "Task: decide whether a paper reports an ACTUAL human user study.\n\n"
    "INCLUDE if the title/abstract clearly indicates any of:\n"
    "- human participants/users were recruited\n"
    "- user study / usability study / human-subject study\n"
    "- survey, interview, diary study, field study, lab study\n"
    "- controlled experiment with participants\n"
    "- mentions N=..., participant demographics, study procedure\n\n"
    "EXCLUDE if:\n"
    "- purely technical paper (system/model/algorithm/benchmark)\n"
    "- evaluation only on datasets/benchmarks/simulations/logs without human subjects\n"
    "- 'user' is mentioned only conceptually with no user study conducted\n\n"
    "Choose 'unsure' ONLY if there is genuinely insufficient information.\n"
    "Be decisive: prefer include/exclude whenever there is evidence.\n\n"
    "Respond ONLY in valid JSON:\n"
    '{ "decision": "include" | "exclude" | "unsure", "confidence": 0.0-1.0 }'
)

def _parse_json_maybe(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {"decision": "unsure", "confidence": 0.0}

def judge_paper(title: str, abstract: str) -> dict:
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": f"Title: {title}\n\nAbstract:\n{abstract}"},
        ],
    )
    content = resp.choices[0].message.content or ""
    out = _parse_json_maybe(content)

    # Normalize/validate
    decision = str(out.get("decision", "unsure")).lower().strip()
    if decision not in {"include", "exclude", "unsure"}:
        decision = "unsure"

    try:
        conf = float(out.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    conf = max(0.0, min(1.0, conf))

    return {"decision": decision, "confidence": conf}



def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required_cols = {"Venue", "Year", "Title", "Abstract", "URL"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")


    if os.path.exists(OUTPUT_FILE):
        existing_df = pd.read_csv(OUTPUT_FILE)
        judged_urls = set(existing_df["URL"].dropna().astype(str))
        print(f"[INFO] Found existing output. Already judged: {len(judged_urls)}")
    else:
        existing_df = pd.DataFrame(columns=list(df.columns) + ["LLM_Decision", "Confidence"])
        judged_urls = set()
        print("[INFO] No existing output found. Starting fresh.")

    new_rows = []
    total = len(df)
    to_process = sum(1 for u in df["URL"].astype(str) if u not in judged_urls)
    print(f"[INFO] Input rows: {total} | To judge (new): {to_process}")

    processed = 0
    for i, row in df.iterrows():
        url = str(row.get("URL", "") or "")
        if not url or url in judged_urls:
            continue

        title = str(row.get("Title", "") or "")
        abstract = str(row.get("Abstract", "") or "")

        try:
            out = judge_paper(title, abstract)
        except Exception as e:
            out = {"decision": "unsure", "confidence": 0.0}
            print(f"[WARN] Judge failed for URL={url[:60]}... Error: {type(e).__name__}: {e}")

        new_rows.append({
            "Venue": row.get("Venue"),
            "Year": row.get("Year"),
            "Title": title,
            "Abstract": abstract,
            "URL": url,
            "LLM_Decision": out["decision"],
            "Confidence": out["confidence"],
        })

        processed += 1
        print(f"[PROGRESS] {processed}/{to_process} -> {out['decision']} (conf={out['confidence']:.2f})")

        # Autosave periodically
        if processed % SAVE_EVERY == 0:
            temp_df = pd.concat([existing_df, pd.DataFrame(new_rows)], ignore_index=True)
            temp_df = temp_df.drop_duplicates(subset=["URL"])
            temp_df.to_csv(OUTPUT_FILE, index=False)
            print(f"[INFO] Autosaved after {processed} new judgments -> {OUTPUT_FILE}")

    # save
    final_df = pd.concat([existing_df, pd.DataFrame(new_rows)], ignore_index=True)
    final_df = final_df.drop_duplicates(subset=["URL"])
    final_df.to_csv(OUTPUT_FILE, index=False)


    print("\n[DONE] Screening complete.")
    print(f"[INFO] New judged: {processed}")
    print(f"[INFO] Total in output (deduped): {len(final_df)}")
    if "LLM_Decision" in final_df.columns:
        print("\n[SUMMARY] Decision counts:")
        print(final_df["LLM_Decision"].value_counts(dropna=False))

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()