"""
Merge CLadder Questions - Add explicit questions to cladder_600.csv

This script matches the IDs in cladder_600.csv with the original
cladder-v1-q-balanced.json to extract the explicit 'question' field
that was missing from the HuggingFace CSV version.
"""

import hashlib
import json
import pandas as pd
from pathlib import Path

def main():
    data_dir = Path(__file__).parent.parent / "data"
    
    # Load original CLadder with questions
    json_path = data_dir / "cladder-v1" / "cladder-v1-q-balanced.json"
    print(f"Loading {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        cladder_full = json.load(f)
    
    # Build lookup by question_id
    id_to_data = {}
    for sample in cladder_full:
        qid = sample.get("question_id")
        if qid is not None:
            id_to_data[qid] = {
                "question": sample.get("question", ""),
                "given_info": sample.get("given_info", "")
            }
    
    print(f"Built lookup with {len(id_to_data)} entries")
    
    # Load our cladder_600.csv
    csv_path = data_dir / "splits" / "openai_subsets" / "cladder_600.csv"
    df = pd.read_csv(csv_path)
    print(f"Loaded cladder_600.csv with {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    
    # Fail hard on mismatches - ensures broken assumptions are caught immediately
    our_ids = df["id"].tolist()
    missing = [qid for qid in our_ids if qid not in id_to_data]
    if missing:
        raise ValueError(f"{len(missing)} IDs missing from original CLADDER test: {missing[:5]}")
    print(f"All {len(our_ids)} IDs matched successfully")
    
    # Add question and given_info columns
    questions = []
    given_infos = []
    for idx, row in df.iterrows():
        qid = row["id"]
        questions.append(id_to_data[qid]["question"])
        given_infos.append(id_to_data[qid]["given_info"])
    
    df["question"] = questions
    df["given_info"] = given_infos
    
    # Reorder columns to put question near the front
    cols = df.columns.tolist()
    new_order = ["id", "prompt", "question", "given_info", "label", "reasoning", 
                 "graph_id", "story_id", "rung", "query_type", "formal_form"]
    new_order = [c for c in new_order if c in cols]
    remaining = [c for c in cols if c not in new_order]
    df = df[new_order + remaining]
    
    # Save updated CSV
    output_path = data_dir / "splits" / "openai_subsets" / "cladder_600.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved updated CSV to {output_path}")
    
    # Emit checksum manifest for ICML reproducibility
    md5 = hashlib.md5(output_path.read_bytes()).hexdigest()
    checksum_path = output_path.parent / "cladder_600.md5"
    checksum_path.write_text(md5 + "\n")
    print(f"Wrote MD5 checksum to {checksum_path}: {md5}")
    
    # Also create a backup
    backup_path = data_dir / "splits" / "openai_subsets" / "cladder_600_with_questions.csv"
    df.to_csv(backup_path, index=False)
    print(f"Saved backup to {backup_path}")
    
    # Show some examples
    print("\n--- Sample Questions ---")
    for i in range(min(3, len(df))):
        row = df.iloc[i]
        print(f"\nID: {row['id']}")
        print(f"Question: {row['question']}")
        print(f"Label: {row['label']}")


if __name__ == "__main__":
    main()
