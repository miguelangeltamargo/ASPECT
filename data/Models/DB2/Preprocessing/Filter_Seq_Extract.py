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

OFFSET = 512  # Define the offset of nt upstream and downstream
LOW_INCLUSION_FLAG = False     # Flag to create modertate Cassettes or not

# Input Files
input_1 = "Con0.tsv"   # Constitutive exons
input_2 = "Cas90.tsv"       # Cassette exons
bed_filename = "hexevent_data.bed"  # Combined BED file
fasta = "hg38.fa"              # Reference FASTA file

# Output Path
dataset_path = f"../datasets/{OFFSET}_Split/"
os.makedirs(dataset_path, exist_ok=True)
# Output File
dataset_filename = f"../datasets/{OFFSET}_Split/{OFFSET}_Split_C_C.csv"

# -----------------------------------------------------------------------------------
# 📌 **Function: Combine Data to BED File with Filtration**
# -----------------------------------------------------------------------------------

def combine_data_to_bed():
    """
    Combine data from constitutive.tsv and cassette.tsv into a BED file with offset.
    Applies filtration based on provided guidelines.
    """
    print("🔄 Combining and filtering data to BED file...")
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

    # Convert relevant columns to numeric, coerce errors to NaN
    numeric_cols1 = ["start", "end", "count", "alt3", "alt5", "alt3+5", "inclLevel", "skip"]
    numeric_cols2 = ["start", "end", "count", "alt3", "alt5", "alt3+5", "skip", "inclLevel", "3usageLevel", "5usageLevel"]

    for col in numeric_cols1:
        data1[col] = pd.to_numeric(data1[col], errors='coerce')
    for col in numeric_cols2:
        data2[col] = pd.to_numeric(data2[col], errors='coerce')

    # Calculate Exon Length for both datasets
    data1['exon_length'] = data1['end'] - data1['start']
    data2['exon_length'] = data2['end'] - data2['start']

    # -----------------------------------------------------------------------------------
    # 📌 **Apply Filtration to Constitutive Exons**
    # -----------------------------------------------------------------------------------
    print("📋 Filtering Constitutive Exons...")
    constitutive_filtered = data1[
        (data1['inclLevel'] >= 1.0) &
        (data1['count'] >= 20) &
        (data1['skip'] <= 1) &
        (data1['alt3'] == 0) &
        (data1['alt5'] == 0) &
        (data1['alt3+5'] == 0) &
        (data1['exon_length'] >= 25)
    ].copy()

    print(f"✅ Constitutive Exons passed: {constitutive_filtered.shape[0]}/{data1.shape[0]}")

    # -----------------------------------------------------------------------------------
    # 📌 **Apply Filtration to Cassette Exons**
    # -----------------------------------------------------------------------------------
    print("📋 Filtering Cassette Exons...")

    # Define inclusion level categories
    low_incl = data2['inclLevel'] <= 0.15
    moderate_incl = (data2['inclLevel'] > 0.15) & (data2['inclLevel'] <= 0.90)
    
    # Apply filters for both low and moderate inclusion levels
    cassette_filtered_low = data2[
        low_incl &
        (data2['skip'] >= 20) &  # Assuming 'high skipping' means at least 20 skip
        (data2['count'] < data2['skip']) &  # Have a stricter filter on having more est of skipped than inclusion
        (data2['alt3'] + data2['alt5'] + data2['alt3+5'] <= 5) &  # Minimal alternative splicing
        (data2['exon_length'] >= 25)
    ].copy()

    cassette_filtered_moderate = data2[
        moderate_incl &
        (data2['skip'] >= 20) &  # Assuming 'high skipping' means at least 20 skip
        (data2['count'] < data2['skip']) &  # Have a stricter filter on having more est of skipped than inclusion
        (data2['alt3'] + data2['alt5'] + data2['alt3+5'] <= 5) &  # Minimal alternative splicing
        (data2['exon_length'] >= 25)
    ].copy()

    # Combine low and moderate inclusion cassette exons
    cassette_filtered = pd.concat([cassette_filtered_low, cassette_filtered_moderate], ignore_index=True)
    
    # # Apply filters for both low and moderate inclusion levels
    # if LOW_INCLUSION_FLAG:
    #     cassette_filtered = data2[
    #         (data2['inclLevel'] <= 0.15) &
    #         (data2['skip'] >= 20) &  # Assuming 'high skipping' means at least 20 skip
    #         (data2['alt3'] + data2['alt5'] + data2['alt3+5'] <= 5) &  # Minimal alternative splicing
    #         (data2['3usageLevel'] == 0.0) &
    #         (data2['5usageLevel'] == 0.0) &
    #         (data2['exon_length'] >= 25)
    #     ].copy()
    # else:
    #     cassette_filtered = data2[
    #         (data2['inclLevel'] > 0.15) & (data2['inclLevel'] <= 0.90) &
    #         (data2['skip'] >= 20) &  # Assuming 'high skipping' means at least 20 skip
    #         (data2['alt3'] + data2['alt5'] + data2['alt3+5'] <= 5) &  # Minimal alternative splicing
    #         (data2['3usageLevel'] == 0.0) &
    #         (data2['5usageLevel'] == 0.0) &
    #         (data2['exon_length'] >= 25)
    #     ].copy()

    print(f"✅ Cassette Exons passed: {cassette_filtered.shape[0]}/{data2.shape[0]}")

    # -----------------------------------------------------------------------------------
    # 📌 **Prepare BED DataFrames with Offset**
    # -----------------------------------------------------------------------------------
    # Process constitutive exons
    bed_data1 = constitutive_filtered[["chromo", "start", "end", "strand"]].copy()
    bed_data1["midpoint"] = ((bed_data1["start"] + bed_data1["end"]) // 2).astype(int)
    bed_data1["start"] = bed_data1["midpoint"] - OFFSET
    bed_data1["end"] = bed_data1["midpoint"] + OFFSET
    bed_data1.drop(columns=["midpoint"], inplace=True)
    bed_data1["label"] = "0"  # Label for constitutive exons

    # Process cassette exons
    bed_data2 = cassette_filtered[["chromo", "start", "end", "strand"]].copy()
    bed_data2["midpoint"] = ((bed_data2["start"] + bed_data2["end"]) // 2).astype(int)
    bed_data2["start"] = bed_data2["midpoint"] - OFFSET
    bed_data2["end"] = bed_data2["midpoint"] + OFFSET
    bed_data2.drop(columns=["midpoint"], inplace=True)
    bed_data2["label"] = "1"  # Label for cassette exons

    # -----------------------------------------------------------------------------------
    # 📌 **Combine Both BED DataFrames**
    # -----------------------------------------------------------------------------------
    combined_bed = pd.concat([bed_data1, bed_data2], ignore_index=True)

    # Ensure BED start is non-negative
    combined_bed['start'] = combined_bed['start'].apply(lambda x: max(x, 0))

    # Reorder columns to standard BED format: chrom, start, end, name, score, strand
    # We'll use 'label' as the name field and set score to 0
    combined_bed = combined_bed[['chromo', 'start', 'end', 'label', 'strand']]
    combined_bed['0'] = 0  # BED format requires a score field; setting to 0

    # Save combined BED data to file
    try:
        combined_bed.to_csv(bed_filename, sep="\t", header=False, index=False)
    except Exception as e:
        print(f"❌ Error writing BED file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"✅ BED file saved as {bed_filename}\n")

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
        labels = bed_df.iloc[:, 3].astype(str).tolist()  # Assuming label is in the 4th column
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

    print(f"\n✅ Extracted sequences saved to {dataset_filename}")

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
