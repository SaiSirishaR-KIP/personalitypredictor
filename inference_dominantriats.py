import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import os
import pickle

#################################
# 1) Define the ImprovedNN model
#################################
class ImprovedNN(nn.Module):
    def __init__(self, input_dim):
        super(ImprovedNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

###############################
# 2) Load Models Dynamically
###############################
checkpoint_dir = './checkpoints'
trait_columns = ['EXT', 'AGR', 'EST', 'CSN', 'OPN']
input_dim = 5  # Because we'll aggregate 50 questions into 5 trait scores

models = {}
for trait in trait_columns:
    checkpoint_path = os.path.join(checkpoint_dir, f'{trait}_best_model.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint for {trait} not found at {checkpoint_path}")

    model = ImprovedNN(input_dim=input_dim)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()
    models[trait] = model

###############################
# 3) Load the Scaler
###############################
scaler_path = os.path.join(checkpoint_dir, 'scaler.pkl')
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Scaler not found at {scaler_path}")

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

###############################
# 4) Trait Mapping
###############################
trait_mapping = {
    "I am the life of the party": "EXT1",
    "I don't talk a lot": "EXT2",
    "I feel comfortable around people": "EXT3",
    "I keep in the background": "EXT4",
    "I start conversations": "EXT5",
    "I have little to say": "EXT6",
    "I talk to a lot of different people at parties": "EXT7",
    "I don't like to draw attention to myself": "EXT8",
    "I don't mind being the center of attention.": "EXT9",
    "I am quiet around strangers": "EXT10",
    "I feel little concern for others": "AGR1",
    "I am interested in people": "AGR2",
    "I insult people": "AGR3",
    "I sympathize with others' feelings": "AGR4",
    "I am not interested in other people's problems": "AGR5",
    "I have a soft heart": "AGR6",
    "I am not really interested in others": "AGR7",
    "I take time out for others": "AGR8",
    "I feel others' emotions": "AGR9",
    "I make people feel at ease": "AGR10",
    "I am always prepared": "CSN1",
    "I leave my belongings around": "CSN2",
    "I pay attention to details": "CSN3",
    "I make a mess of things.": "CSN4",
    "I get chores done right away": "CSN5",
    "I often forget to put things back in their proper place": "CSN6",
    "I like order": "CSN7",
    "I shirk my duties": "CSN8",
    "I follow a schedule": "CSN9",
    "I am exacting in my work": "CSN10",
    "I get stressed out easily": "EST1",
    "I am relaxed most of the time": "EST2",
    "I worry about things": "EST3",
    "I seldom feel blue": "EST4",
    "I am easily disturbed": "EST5",
    "I get upset easily": "EST6",
    "I change my mood a lot": "EST7",
    "I have frequent mood swings": "EST8",
    "I get irritated easily": "EST9",
    "I often feel blue": "EST10",
    "I have a rich vocabulary": "OPN1",
    "I have difficulty understanding abstract ideas": "OPN2",
    "I have a vivid imagination": "OPN3",
    "I am not interested in abstract ideas": "OPN4",
    "I have excellent ideas": "OPN5",
    "I do not have a good imagination": "OPN6",
    "I am quick to understand things": "OPN7",
    "I use difficult words": "OPN8",
    "I spend time reflecting on things": "OPN9",
    "I am full of ideas": "OPN10"
}

#######################################
# 5) Main Inference Function
#######################################
def predict_personality(data):
    """
    Predict personality traits from raw 50-question survey data.

    Steps:
      1) Map question text -> trait item (EXT1, EXT2, etc.)
      2) Reverse-score negative items
      3) Aggregate to 5 columns: EXT, AGR, EST, CSN, OPN
      4) Scale, run inference on each trait model
      5) Return a DataFrame of predictions, plus top-2 dominant traits, plus the processed data

    Args:
        data (pd.DataFrame): A DataFrame with the survey questions as columns, e.g.:
          {
             "I am the life of the party": 5,
             "I don't talk a lot": 2,
             ...
             "I am full of ideas": 4
          }

    Returns:
        predictions_df: pd.DataFrame with columns [EXT, AGR, EST, CSN, OPN] => predicted scores
        dominant_traits_list: list of top-2 traits for each row
        processed_data: the post-aggregation DataFrame with columns [EXT, AGR, EST, CSN, OPN]
    """

    # 1) Rename columns from question text -> trait item codes (EXT1, EXT2, etc.)
    mapped_columns = {}
    for col in data.columns:
        if col in trait_mapping:
            mapped_columns[col] = trait_mapping[col]

    data = data.rename(columns=mapped_columns)

    # 2) Reverse-scoring
    negatively_keyed = [
        'EXT2', 'EXT4', 'EXT6', 'EXT8', 'EXT10',
        'EST2', 'EST4',
        'AGR1', 'AGR3', 'AGR5', 'AGR7',
        'CSN2', 'CSN4', 'CSN6', 'CSN8',
        'OPN2', 'OPN4', 'OPN6'
    ]
    for col in negatively_keyed:
        if col in data.columns:
            data[col] = 6 - data[col]

    # 3) Ensure all 50 sub-trait columns exist; fill missing with 3 (neutral).
    all_trait_columns = [f"{trait}{i}" for trait in ['EXT','AGR','EST','CSN','OPN'] for i in range(1,11)]
    for col in all_trait_columns:
        if col not in data.columns:
            data[col] = 3  # default score if missing

    # 4) Aggregate each Big Five trait
    features = ['EXT', 'AGR', 'EST', 'CSN', 'OPN']
    for trait in features:
        sub_cols = [f"{trait}{i}" for i in range(1,11)]
        data[trait] = data[sub_cols].mean(axis=1)

    # 5) Prepare data for model inference
    # We'll scale the aggregated columns [EXT, AGR, EST, CSN, OPN].
    X_vals = data[features].values
    X_scaled = scaler.transform(X_vals)
    X_torch = torch.tensor(X_scaled, dtype=torch.float32)

    # 6) Predict for each trait
    predictions = {}
    for trait, model in models.items():
        with torch.no_grad():
            raw_preds = model(X_torch).numpy().flatten()
            predictions[trait] = raw_preds  # shape: (num_rows,)

    # 7) Convert predictions to DataFrame
    predictions_df = pd.DataFrame(predictions)

    # 8) Identify top 2 dominant traits per row
    #    Sort each row's predictions descending & grab the top 2 columns
    dominant_traits_series = predictions_df.apply(lambda row: row.nlargest(2).index.tolist(), axis=1)
    dominant_traits_list = dominant_traits_series.tolist()

    return predictions_df, dominant_traits_list, data
