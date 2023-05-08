import numpy as np
import pandas as pd

def z_score_normalize(scores):
    """
    Normalize a list of scores between 0 and 1 using z-score normalization.
    
    Args:
        scores (list): List of scores to be normalized.
    
    Returns:
        list: Normalized scores.
    """
    # Convert the list of scores to a numpy array
    scores_array = np.array(scores)
    
    # Calculate the mean and standard deviation of the scores
    mean_score = np.mean(scores_array)
    std_score = np.std(scores_array)
    
    # Normalize the scores using z-score normalization
    normalized_scores = (scores_array - mean_score) / std_score
    
    # Scale the normalized scores to be between 0 and 1
    normalized_scores = (normalized_scores - np.min(normalized_scores)) / (np.max(normalized_scores) - np.min(normalized_scores))
    
    return normalized_scores.tolist()

df = pd.read_csv('dataset/db_with_inserted_textual_mistakes_new.csv')
df['normalized_gpt_score'] = z_score_normalize(df['gpt_sim_score'])
df['normalized_gpt2_score'] = z_score_normalize(df['gpt2_sim_score'])
#print(df['normalized_gpt2_score'])
df.to_csv('dataset/db_with_inserted_textual_mistake_new.csv')