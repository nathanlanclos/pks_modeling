"""
Created on Tue Dec 12 2024
This script generates all the blosum based features, works with .csv/.pkl. Call in notebooks
@author: Nathan
"""
import pandas as pd

def process_file_with_blosum(input_file, output_file, blosum_matrix):
    """
    Adds BLOSUM score and BLOSUM embeddings to a DataFrame from a CSV or pickle file.

    Parameters:
        input_file (str): Path to the input CSV or pickle file.
        output_file (str): Path to save the output file.
        blosum_matrix (dict): BLOSUM matrix to calculate scores and embeddings.

    Returns:
        pd.DataFrame: DataFrame with BLOSUM score and embeddings added.
    """
    # Load the input file
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    elif input_file.endswith('.pkl'):
        df = pd.read_pickle(input_file)
    else:
        raise ValueError("Unsupported file format. Only .csv and .pkl are supported.")

    # Ensure the file has a 'sequence' column
    if 'sequence' not in df.columns:
        raise ValueError("The input file must contain a 'sequence' column.")

    # Define a function to calculate the BLOSUM score for a sequence
    def calculate_blosum_score(sequence, matrix):
        score = 0
        for i in range(len(sequence) - 1):
            score += matrix.get((sequence[i], sequence[i+1]), 0)
        return score

    # Define a function to calculate BLOSUM embeddings for a sequence
    def calculate_blosum62_embedding(sequence):
        embedding = []
        for aa in sequence:
            embedding.append(blosum_matrix.get((aa, aa), 0))  # Default score = 0
        return embedding

    # Add BLOSUM score column
    df['blosum_score'] = df['sequence'].apply(lambda seq: calculate_blosum_score(seq, blosum_matrix))

    # Add BLOSUM embedding column
    df['blosum62_embedding'] = df['sequence'].apply(calculate_blosum62_embedding)

    # Save the updated DataFrame
    if output_file.endswith('.csv'):
        df.to_csv(output_file, index=False)
    elif output_file.endswith('.pkl'):
        df.to_pickle(output_file)
    else:
        raise ValueError("Unsupported output file format. Only .csv and .pkl are supported.")

    return df

# Example Usage
if __name__ == "__main__":
    # Manually define the BLOSUM62 matrix
    blosum62 = {
        ('A', 'A'): 4, ('R', 'R'): 5, ('N', 'N'): 6, ('D', 'D'): 6, ('C', 'C'): 9,
        ('Q', 'Q'): 5, ('E', 'E'): 5, ('G', 'G'): 6, ('H', 'H'): 8, ('I', 'I'): 4,
        ('L', 'L'): 4, ('K', 'K'): 5, ('M', 'M'): 5, ('F', 'F'): 6, ('P', 'P'): 7,
        ('S', 'S'): 4, ('T', 'T'): 5, ('W', 'W'): 11, ('Y', 'Y'): 7, ('V', 'V'): 4
    }

    input_path = "input_file.csv"  # Change to your input file path
    output_path = "output_file.csv"  # Change to your output file path

    updated_df = process_file_with_blosum(input_path, output_path, blosum62)
    print("Updated file saved to:", output_path)
