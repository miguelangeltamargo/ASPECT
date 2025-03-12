import pandas as pd


def convert_file_labels(inputfile):
    # Read the CSV file
    # Assuming the data is comma-separated and does not contain a header
    column_names = ["label", "sequence"]
    df = pd.read_csv(inputfile, names=column_names)

    # Convert labels with error handling
    df["label"] = df["label"].map({"constitutive": 0, "alt_five": 1})

    # Fill NaN values with a default (e.g., -1) to avoid conversion issues
    df["label"].fillna(-1, inplace=True)

    df["label"] = df["label"].astype(int)


    # Save to a new file
    df.to_csv(inputfile, index=False)

    print(f"Processed data saved to {inputfile}")


# convert_file_labels("../datasets/512_Split/TeamsShare/Reduced_Inclusion_100/cons_8000_alt5/train.csv")
# convert_file_labels("../datasets/512_Split/TeamsShare/Reduced_Inclusion_100/cons_8000_alt5/dev.csv")
# convert_file_labels("../datasets/512_Split/TeamsShare/Reduced_Inclusion_100/cons_8000_alt5/test.csv")

convert_file_labels("../datasets/512_Split/TeamsShare/Data/512_split_1_cons_alt5/train.csv")
convert_file_labels("../datasets/512_Split/TeamsShare/Data/512_split_1_cons_alt5/dev.csv")
convert_file_labels("../datasets/512_Split/TeamsShare/Data/512_split_1_cons_alt5/test.csv")
