import pybedtools

# Specify the chromosome, start position, and end position
chromosome = 'chr1'
start = 1000
end = 2200

# Create a BedTool interval from the specified location
interval = pybedtools.BedTool('{}\t{}\t{}'.format(chromosome, start, end), from_string=True)

# Specify the path to the FASTA file
fasta_file = 'hg38C.fa'

# Extract the sequence from the FASTA file using the interval
extracted_sequence = interval.sequence(fi=fasta_file)

# Print the extracted sequence
print(extracted_sequence.seq(interval, fasta_file))
