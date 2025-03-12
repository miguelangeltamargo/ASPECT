import pandas as pd
import numpy as np
import pybedtools
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import sys
import time
import os

# -----------------------------------------------------------------------------------
# 📌 **Constants and File Paths**
# -----------------------------------------------------------------------------------

OFFSET = 1024  # Define the offset for 500 nt upstream and downstream

# Input Files
input_1 = "constitutive.tsv"   # Constitutive exons
input_2 = "cassette.tsv"       # Cassette exons
bed_filename = "hexevent_data.bed"  # Combined BED file
fasta = "hg38.fa"              # Reference FASTA file

# Output File
dataset_filename = f"../datasets/{OFFSET}_Split_C_C.csv"

# -----------------------------------------------------------------------------------
# 📌 **Function: Combine Data to BED File**
# -----------------------------------------------------------------------------------

def combine_data_to_bed():
    """
    Combine data from constitutive.tsv and cassette.tsv into a BED file with offset.
    """
    print("🔄 Combining data to BED file...")
    # Read input TSV files
    try:
        data1 = pd.read_csv(input_1, sep="\t", low_memory=False)
        data2 = pd.read_csv(input_2, sep="\t", low_memory=False)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Replace missing values represented by '#' with NaN
    data1.replace('#', np.nan, inplace=True)
    data2.replace('#', np.nan, inplace=True)

    # Process constitutive exons (data1)
    bed_data1 = data1[["chromo", "start", "end", "strand"]].copy()
    bed_data1["midpoint"] = (bed_data1["start"] + bed_data1["end"]) // 2
    bed_data1["start"] = bed_data1["midpoint"] - OFFSET
    bed_data1["end"] = bed_data1["midpoint"] + OFFSET
    bed_data1.drop(columns=["midpoint"], inplace=True)
    bed_data1["label"] = "0"  # Label for constitutive exons

    # Process cassette exons (data2)
    bed_data2 = data2[["chromo", "start", "end", "strand"]].copy()
    bed_data2["midpoint"] = (bed_data2["start"] + bed_data2["end"]) // 2
    bed_data2["start"] = bed_data2["midpoint"] - OFFSET
    bed_data2["end"] = bed_data2["midpoint"] + OFFSET
    bed_data2.drop(columns=["midpoint"], inplace=True)
    bed_data2["label"] = "1"  # Label for cassette exons

    # Combine both BED DataFrames
    combined_bed = pd.concat([bed_data1, bed_data2], ignore_index=True)

    # Save combined BED data to file
    try:
        combined_bed.to_csv(bed_filename, sep="\t", header=False, index=False)
    except Exception as e:
        print(f"❌ Error writing BED file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"✅ BED file saved as `{bed_filename}`\n")

# -----------------------------------------------------------------------------------
# 📌 **Function: Process a Single Sequence Line**
# -----------------------------------------------------------------------------------

def process_sequence(line_info, labels):
    """
    Process a single line from the sequences file.

    Args:
        line_info (tuple): (line_number, line_content)
        labels (list): List of labels corresponding to each BED interval.

    Returns:
        tuple or None: (sequence, label) or None if header
    """
    try:
        line_number, line = line_info
        if line.startswith(">"):
            # Header line; skip
            return None
        else:
            # Sequence line
            sequence = line.strip().upper()
            label_idx = line_number // 2  # Assuming each sequence follows a header
            if label_idx < len(labels):
                label = labels[label_idx]
                return (sequence, label)
            else:
                print(f"⚠️ Warning: line_number {line_number} exceeds labels length.", file=sys.stderr)
                return None
    except Exception as e:
        print(f"❌ Error processing line {line_number}: {e}", file=sys.stderr)
        return None

# -----------------------------------------------------------------------------------
# 📌 **Function: Extract Sequences from FASTA**
# -----------------------------------------------------------------------------------

def extract_sequences_from_fa():
    """
    Extract sequences from FASTA using BED intervals and save to CSV.
    Utilizes parallel processing with joblib, progress bars, timing, and output validation.
    """
    start_time = time.time()
    print("🚀 Starting sequence extraction...\n")

    # Convert BED to BedTool object and extract sequences
    try:
        bed_intervals = pybedtools.BedTool(bed_filename)
        sequences = bed_intervals.sequence(fi=fasta, s=True)
    except Exception as e:
        print(f"❌ Error with pybedtools: {e}", file=sys.stderr)
        sys.exit(1)

    # Read BED file labels
    try:
        bed_df = pd.read_csv(bed_filename, sep="\t", header=None)
        labels = bed_df.iloc[:, -1].tolist()
    except Exception as e:
        print(f"❌ Error reading BED file: {e}", file=sys.stderr)
        sys.exit(1)

    total_labels = len(labels)
    print(f"📊 Total labels (BED intervals): {total_labels}")

    # Read sequences file
    try:
        with open(sequences.seqfn, 'r') as seq_file:
            lines = list(enumerate(seq_file))
    except Exception as e:
        print(f"❌ Error reading sequences file: {e}", file=sys.stderr)
        sys.exit(1)

    total_lines = len(lines)
    print(f"📄 Total lines in sequences file: {total_lines}\n")

    # Validation: Expected number of records
    expected_records = total_labels
    print(f"🔍 Expected number of records (sequences): {expected_records}\n")

    # Set up joblib Parallel processing with tqdm progress bar
    num_cores = os.cpu_count() or 1
    print(f"💻 Using {num_cores} CPU cores for processing.\n")

    # Initialize tqdm_joblib to integrate joblib with tqdm
    with tqdm_joblib(tqdm(total=total_lines, desc="Processing sequences")):
        # Parallel processing: Process all lines
        records = Parallel(n_jobs=num_cores)(
            delayed(process_sequence)(line_info, labels) for line_info in lines
        )

    # Filter out None values (headers)
    records = [record for record in records if record]

    # Save to CSV
    try:
        seq_df = pd.DataFrame(records, columns=["sequence", "label"])
        seq_df.to_csv(dataset_filename, index=False)
    except Exception as e:
        print(f"❌ Error writing CSV file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\n✅ Extracted sequences saved to `{dataset_filename}`")

    # Output Validation
    actual_records = len(seq_df)
    print(f"📈 Expected records: {expected_records}")
    print(f"📉 Actual records: {actual_records}")
    if actual_records != expected_records:
        print(f"⚠️ WARNING: Mismatch in records! Expected {expected_records}, but got {actual_records}", file=sys.stderr)
    else:
        print("✅ Output validation passed: Record counts match.")

    # Timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"\n⏱️ Total processing time: {int(minutes)}m {seconds:.2f}s\n")

# -----------------------------------------------------------------------------------
# 📌 **Main Function**
# -----------------------------------------------------------------------------------

def main():
    """
    Main function to execute the script.
    """
    total_start_time = time.time()

    combine_data_to_bed()
    extract_sequences_from_fa()

    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    minutes, seconds = divmod(total_elapsed_time, 60)
    print(f"🔚 Total script execution time: {int(minutes)}m {seconds:.2f}s")

# -----------------------------------------------------------------------------------
# 📌 **Entry Point**
# -----------------------------------------------------------------------------------

if __name__ == "__main__":
    main()