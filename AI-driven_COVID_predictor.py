import subprocess
import sys

# List of required packages
required_packages = [
    'pandas',
    'numpy',
    'tensorflow',
    'biopython',
    'scikit-learn',
    'matplotlib'
]

# Check and install packages
def install(package):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"{package} is not installed. Installing...")
        install(package)
    else:
        print(f"{package} is already installed.")

import os 
import pandas as pd
import numpy as np
import tensorflow as tf
from Bio.Align import PairwiseAligner
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from datetime import datetime
from Bio.Align import PairwiseAligner

# Download the trained model
if not os.path.exists('model_trained.h5'):
    subprocess.run(['wget', '-O', 'model_trained.h5', 'https://github.com/caiocheohen/ml_training/blob/main/model_trained.h5?raw=true'], check=True)

# Load the trained model
model = load_model('model_trained.h5')

# Extract max_seq_length from model
max_seq_length = model.layers[0].input_shape[1] 

# Training Code Functions

# Define the secondary structure information
sec_struc = {'H': [1, 0, 0], 'B': [0, 1, 0], 'C': [0, 0, 1]}

# Define the amino acid properties
aa_props = {'A': [0.62, -0.5, 1.8, 0, 0, 1, 1, 0, 0, 0],
            'C': [0.29, -1, 2.5, 0, 0, 1, 0, 1, 0, 0],
            'D': [-0.9, 3, -3.5, 0, 1, 0, 0, 0, 0, 0],
            'E': [-0.74, 3, -3.5, 0, 1, 0, 0, 0, 0, 0],
            'F': [1.19, -2.5, 2.8, 0, 0, 1, 1, 0, 0, 0],
            'G': [0.48, 0, -0.4, 1, 0, 0, 0, 0, 0, 1],
            'H': [-0.4, -0.5, -3.2, 0, 0, 1, 0, 0, 1, 0],
            'I': [1.38, -1.8, 4.5, 0, 0, 1, 1, 0, 0, 0],
            'K': [-1.5, 3, -3.9, 0, 1, 0, 0, 0, 0, 0],
            'L': [1.06, -1.8, 3.8, 0, 0, 1, 1, 0, 0, 0],
            'M': [0.64, -1.3, 1.9, 0, 0, 1, 1, 1, 0, 0],
            'N': [-0.78, 2, -3.5, 0, 1, 0, 0, 0, 0, 0],
            'P': [0.12, 0, -1.6, 0, 0, 0, 0, 0, 0, 0],
            'Q': [-0.85, 2, -3.5, 0, 1, 0, 0, 0, 1, 0],
            'R': [-2.53, 3, -4.5, 0, 1, 0, 0, 0, 1, 0],
            'S': [-0.18, 0.3, -0.8, 1, 0, 0, 0, 0, 0, 1],
            'T': [-0.05, -0.4, -0.7, 1, 0, 1, 0, 0, 0, 1],
            'V': [1.08, -1.5, 4.2, 0, 0, 1, 1, 0, 0, 0],
            'W': [0.81, -3.4, -0.9, 0, 0, 1, 1, 0, 0, 0],
            'Y': [0.26, -2.3, -1.3, 0, 0, 1, 1, 0, 0, 0]}

def fasta_to_numeric(seq):
    # Fixed size of number sequence
    FIXED_SIZE = 3013

    if seq.startswith('>'):
        seq = seq.split('\n', 1)[1].replace('\n', '')
    else:
        seq = seq.replace('\n', '')

    amino_acids = list(seq)
    numeric_seq = []
    hbond_aa_freq = []
    for aa in ['S', 'T', 'N', 'Q', 'H', 'Y']:
        hbond_aa_freq.append(amino_acids.count(aa) / len(amino_acids))

    for i, aa in enumerate(amino_acids):
        if RBD_START <= i <= RBD_END:
            penalty = 5
        else:
            penalty = 1
        if aa in aa_props:
            numeric_seq.append([prop * penalty for prop in aa_props[aa]])
        else:
            numeric_seq.append([0] * 10)
        if RBD_START <= i <= RBD_END:
            sec_struc_aa = 'H'
        else:
            sec_struc_aa = 'C'
        if sec_struc_aa in sec_struc:
            numeric_seq.append(sec_struc[sec_struc_aa])
        else:
            numeric_seq.append([0, 0, 0])

    numeric_seq.append(hbond_aa_freq)
    numeric_seq = np.concatenate(numeric_seq)

    # Fill the number sequence with zeros if necessary
    if len(numeric_seq) < FIXED_SIZE:
        padding = np.zeros(FIXED_SIZE - len(numeric_seq))
        numeric_seq = np.concatenate((numeric_seq, padding))

    return numeric_seq

def get_amino_acids_percent(seq):
    aa_counts = ProteinAnalysis(seq).count_amino_acids()
    total_aa = sum(aa_counts.values())
    return {aa: count / total_aa for aa, count in aa_counts.items()}

# Define the RBD region and amino acid properties
RBD_START = 319
RBD_END = 542

# Function to calculate the amino acid composition
def compute_aa_composition(seq):
    aa_counts = ProteinAnalysis(seq).count_amino_acids()
    total_aa = sum(aa_counts.values())
    return [count / total_aa for count in aa_counts.values()]

# Function to calculate the length of the sequence
def compute_sequence_length(seq):
    return len(seq)

# Function to calculate amino acid diversity
def compute_aa_diversity(seq):
    return len(set(seq))

# Function to calculate the average hydrophobicity
def compute_hydrophobicity(seq):
    hydropathy_scale = {
        'A': 1.8,
        'R': -4.5,
        'N': -3.5,
        'D': -3.5,
        'C': 2.5,
        'Q': -3.5,
        'E': -3.5,
        'G': -0.4,
        'H': -3.2,
        'I': 4.5,
        'L': 3.8,
        'K': -3.9,
        'M': 1.9,
        'F': 2.8,
        'P': -1.6,
        'S': -0.8,
        'T': -0.7,
        'W': -0.9,
        'Y': -1.3,
        'V': 4.2
    }
    aa_percent = get_amino_acids_percent(seq)
    hydrophobicity = sum(hydropathy_scale[aa] * aa_percent[aa] for aa in hydropathy_scale if aa in aa_percent)
    return hydrophobicity

def compute_net_charge(seq):
    charged_aa = ['D', 'E', 'H', 'K', 'R']
    protein = ProteinAnalysis(seq)
    net_charge = protein.charge_at_pH(7.4)
    return net_charge

# Hopp and Woods polarity (1981)
def compute_polarity(seq):
    polarity_scale = {
        'A': 0.62,
        'R': -1.01,
        'N': -0.60,
        'D': -0.77,
        'C': 0.29,
        'Q': -0.22,
        'E': -0.64,
        'G': 0.00,
        'H': -0.40,
        'I': 1.38,
        'L': 1.06,
        'K': -0.99,
        'M': 0.64,
        'F': 1.19,
        'P': 0.12,
        'S': -0.18,
        'T': -0.05,
        'W': 0.81,
        'Y': 0.26,
        'V': 1.08
    }
    aa_percent = get_amino_acids_percent(seq)
    polarity = sum([polarity_scale[aa] * aa_percent[aa] for aa in polarity_scale if aa in aa_percent])
    return polarity

def calculate_features(data):
    data['aa_composition'] = data['Fastas'].apply(compute_aa_composition)
    data['sequence_length'] = data['Fastas'].apply(compute_sequence_length)
    data['aa_diversity'] = data['Fastas'].apply(compute_aa_diversity)
    data['hydrophobicity'] = data['Fastas'].apply(compute_hydrophobicity)
    data['net_charge'] = data['Fastas'].apply(compute_net_charge)
    data['secondary_structure'] = data['Fastas'].apply(compute_secondary_structure)
    data['polarity'] = data['Fastas'].apply(compute_polarity)
    data['sequence_descriptors'] = data.apply(lambda x: compute_sequence_descriptors(x['Fastas']), axis=1)
    data['sequence_descriptors'] = data['sequence_descriptors'].apply(np.array)
    return data

def convert_sequences_to_numeric(data):
    data['numeric_seq'] = data['Fastas'].apply(fasta_to_numeric)
    # Convert number string to float32
    data['numeric_seq'] = data['numeric_seq'].apply(lambda seq: np.array(seq, dtype=np.float32))
    return data

def compute_sequence_descriptors(seq):
    aa_counts = ProteinAnalysis(seq).count_amino_acids()
    total_aa = sum(aa_counts.values())
    aa_composition = [count / total_aa for count in aa_counts.values()]
    sequence_length = len(seq)
    aa_diversity = len(set(seq))
    hydrophobicity = compute_hydrophobicity(seq)
    net_charge = compute_net_charge(seq)
    secondary_structure = compute_secondary_structure(seq)
    numeric_seq = fasta_to_numeric(seq)
    return aa_composition + [sequence_length, aa_diversity, hydrophobicity, net_charge] + secondary_structure + numeric_seq.tolist()


def compute_secondary_structure(seq):
    helix_count, turn_count, sheet_count = ProteinAnalysis(seq).secondary_structure_fraction()
    return [helix_count, turn_count, sheet_count]

def preprocess_sequences(data):
    # Use max_seq_length extracted from the model
    data['numeric_seq_padded'] = pad_sequences(data['numeric_seq'], maxlen=max_seq_length, dtype='float32').tolist()  
    return data

os.system('clear')

def get_sequence_id():
    # Ask the user if they want to use an ID
    seq_id = input("Would you like to use an ID for the sequence? (Type 'y' for yes or 'n' for no): ").strip().lower()
    
    if seq_id == 'y':
        seq_id = input("Please enter your desired identification: ").strip()
    else:
        seq_id = "results"
    
    return seq_id

def create_results_directory(seq_id):
    # Create a directory to store the results
    if not os.path.exists(seq_id):
        os.makedirs(seq_id)
    return seq_id

# Request identification and create directory
seq_id = get_sequence_id()
results_directory = create_results_directory(seq_id)


# Get the user's protein sequence
seq_protein = input("Enter the protein sequence in FASTA format: ")

# Obtain other user data (with option for "missing information (mi)")
gender = input("Enter gender (M/F/missing information (mi)): ")
age = input("Enter age (or 'missing information (mi)'): ")
clade = input("Enter clade (or 'missing information (mi)'): ")
lineage = input("Enter lineage (or 'missing information (mi)'): ")

# Create a dictionary with user data
user_data = {
    'Fastas': seq_protein,
    'Gender': gender,
    'Age': age,
    'Clade': clade,
    'Lineage': lineage
}

# Create a DataFrame with user data
df_user = pd.DataFrame([user_data])

# Pre-process user data
df_user = calculate_features(df_user)
df_user = convert_sequences_to_numeric(df_user)
df_user = preprocess_sequences(df_user)

# Create input variables for the model
X1_user = np.vstack(df_user['numeric_seq_padded'].to_numpy()).astype(np.float32)

# Create dummy variables for categorical data
X2_user = pd.get_dummies(df_user[['Gender', 'Age', 'Clade', 'Lineage']])

# Adjust the columns of X2_user to match those from training
training_columns = ['Gender_F', 'Gender_M', 'Gender_missing information (mi)', 'Age_18-29', 'Age_30-39', 'Age_40-49', 
                    'Age_50-59', 'Age_60-69', 'Age_70-79', 'Age_80-89', 'Age_90-99', 'Age_missing information (mi)', 
                    'Clade_19A', 'Clade_19B', 'Clade_20A', 'Clade_20B', 'Clade_20C', 'Clade_20I', 
                    'Clade_21A', 'Clade_21I', 'Clade_21J', 'Clade_21K', 'Clade_21L', 'Clade_22A', 
                    'Clade_22B', 'Clade_22C', 'Clade_missing information (mi)', 'Lineage_B.1', 'Lineage_B.1.1', 
                    'Lineage_B.1.1.28', 'Lineage_B.1.1.33', 'Lineage_B.1.1.7', 'Lineage_B.1.177', 
                    'Lineage_B.1.2', 'Lineage_B.1.351', 'Lineage_B.1.427', 'Lineage_B.1.429', 
                    'Lineage_B.1.617.2', 'Lineage_P.1', 'Lineage_P.2', 'Lineage_missing information (mi)']

for column in training_columns:
    if column not in X2_user.columns:
        X2_user[column] = 0

X2_user = X2_user[training_columns]

import numpy as np
from datetime import datetime

# Print the size of X1_user and X2_user
#print("Size of X1_user:", X1_user.shape)
#print("Size of X2_user:", X2_user.shape)

# Concatenate input variables
X_user = np.hstack((X1_user, X2_user.to_numpy()))
#print("Size of X_user after concatenation:", X_user.shape)

# Convert X_user to float32  
X_user = X_user.astype(np.float32)

# Check the size of X_user and adjust
if X_user.shape[1] < 16730:
    #print("X_user is smaller than 16730, padding.")
    X_user = np.pad(X_user, ((0, 0), (0, 16730 - X_user.shape[1])), 'constant')
elif X_user.shape[1] > 16730:
    #print("X_user is larger than 16730, truncating.")
    X_user = X_user[:, :16730]

# Check size after adjustment
print("Size of X_user after adjustment:", X_user.shape)

# Reshape and make the prediction
try:
    prediction_prob = model.predict(X_user.reshape(1, 16730, 1))
except ValueError as e:
    print("Error making the prediction:", e)
    print("Shape of X_user:", X_user.shape)
    raise

# Make the prediction
prediction_class = "severe" if prediction_prob[0][0] < 0.5 else "mild"

# Show the prediction to the user
print("\nPrediction:")
print("Probability of severe:", 1 - prediction_prob[0][0])
print("Probability of mild:", prediction_prob[0][0])
print("Predicted class:", prediction_class)

# Generate the pie chart
labels = ['Severe', 'Mild']
sizes = [1 - prediction_prob[0][0], prediction_prob[0][0]]
colors = ['#ff9999', '#66b3ff']
explode = (0.1, 0)  # Explode the 'Severe' slice

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Ensure the pie chart is drawn as a circle

# Save the chart to an image file
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(os.path.join(results_directory, f'prediction_chart_{current_datetime}.png'))

# Save the information to a text file
with open(os.path.join(results_directory, f'prediction_info_{current_datetime}.txt'), 'w') as f:
    f.write(f'Protein sequence: {seq_protein}\n')
    f.write(f'Gender: {gender}\n')
    f.write(f'Age: {age}\n')
    f.write(f'Clade: {clade}\n')
    f.write(f'Lineage: {lineage}\n')
    f.write('\nPrediction:\n')
    f.write(f'Probability of severe: {1 - prediction_prob[0][0]:.4f}\n')
    f.write(f'Probability of mild: {prediction_prob[0][0]:.4f}\n')
    f.write(f'Predicted class: {prediction_class}\n')

# End the program
sys.exit()  # Add this line to exit the program

