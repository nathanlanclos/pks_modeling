import pandas as pd
import blosum as bl

# Path to data folder
directory = '../GATSol/dataset/'

# Load the DataFrame with embeddings
df = pd.read_pickle(directory + 'eSol_Test.pkl')

# Initialize BLOSUM62 matrix
matrix = bl.BLOSUM(62, default=0)

val = matrix['A']['Y']

# Function to calculate BLOSUM score for a sequence
def calculate_blosum_score(sequence, matrix):
    score = 0
    # Loop over pairs of amino acids in the sequence
    for i in range(len(sequence) - 1):
        score += matrix[sequence[i]][sequence[i+1]]
    return score

# Apply the function to each sequence in the DataFrame
df['blosum_score'] = df['sequence'].apply(lambda seq: calculate_blosum_score(seq, matrix))

# Now `df` has an additional column with BLOSUM scores for each sequence
print(df[['sequence', 'blosum_score']])