import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, signal, optimize, special
import seaborn as sns
import requests
from io import StringIO
import os
from tqdm.notebook import tqdm

# Display settings for better visualization in notebooks
%matplotlib inline
plt.style.use('seaborn-whitegrid')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# ============= DATA LOADING FUNCTIONS =============

def load_species_data(reload=False):
    """
    Load protein data for all species, either from local files or by downloading
    
    Parameters:
    -----------
    reload : bool
        If True, forces redownload even if local files exist
        
    Returns:
    --------
    pandas.DataFrame
        Combined dataset with all species
    """
    # Define URLs and species names
    galaxy_urls = [
        "https://usegalaxy.org/api/datasets/f9cad7b01a47213573dbed9a16a03dc2/display?to_ext=csv",
        "https://usegalaxy.org/api/datasets/f9cad7b01a472135f8160c40120e3811/display?to_ext=csv",
        "https://usegalaxy.org/api/datasets/f9cad7b01a472135f92f314de2216ad8/display?to_ext=csv",
        "https://usegalaxy.org/api/datasets/f9cad7b01a4721355467cd4df3e2393a/display?to_ext=csv",
        "https://usegalaxy.org/api/datasets/f9cad7b01a472135caf4206cbb5e4bbd/display?to_ext=csv",
        "https://usegalaxy.org/api/datasets/f9cad7b01a4721357beb13ad0abffc44/display?to_ext=csv",
        "https://usegalaxy.org/api/datasets/f9cad7b01a4721352084a53c8283a568/display?to_ext=csv",
        "https://usegalaxy.org/api/datasets/f9cad7b01a472135cbc8cacc02760fde/display?to_ext=csv",
        "https://usegalaxy.org/api/datasets/f9cad7b01a4721354c8f13a506a667b0/display?to_ext=csv",
        "https://usegalaxy.org/api/datasets/f9cad7b01a47213579c0af40dc717484/display?to_ext=csv",
        "https://usegalaxy.org/api/datasets/f9cad7b01a4721358914ace52284da51/display?to_ext=csv"
    ]

    species_list = [
        "Human",
        "Mouse",
        "Cow",
        "Orangutan",
        "Pig",
        "Rabbit",
        "Rat",
        "Zebrafish",
        "D. melanogaster",
        "Chicken",
        "Yeast"
    ]
    
    combined_file = "uniprot_All_Species.csv"
    
    # Check if combined file exists and we don't want to reload
    if os.path.exists(combined_file) and not reload:
        print(f"Loading combined dataset from {combined_file}")
        return pd.read_csv(combined_file)
    
    # Check if all individual files exist and we don't want to reload
    all_files_exist = all(os.path.exists(f"uniprot_{species}.csv") for species in species_list)
    if all_files_exist and not reload:
        print("Loading and combining individual species files...")
        all_dfs = []
        for species in species_list:
            df = pd.read_csv(f"uniprot_{species}.csv")
            all_dfs.append(df)
        all_uniprot = pd.concat(all_dfs, ignore_index=True)
        all_uniprot.to_csv(combined_file, index=False)
        return all_uniprot
    
    # Otherwise, download and process data
    print("Downloading and processing protein data...")
    all_dfs = []
    
    for url, species in zip(galaxy_urls, species_list):
        print(f"Processing {species}...")
        
        try:
            # Read the data
            response = requests.get(url)
            response.raise_for_status()
            galaxy_dataset = pd.read_csv(StringIO(response.text))
            
            # Calculate amino acid length and other metrics
            galaxy_dataset['species'] = species
            galaxy_dataset['aa_length'] = galaxy_dataset['oSequence'].str.len()
            galaxy_dataset['gene_size'] = galaxy_dataset['chromEnd'] - galaxy_dataset['chromStart']
            galaxy_dataset['intergenic_distance'] = galaxy_dataset['chromStart'] - galaxy_dataset['chromStart'].shift(1)
            galaxy_dataset['relative_pos'] = galaxy_dataset['chromStart'] / galaxy_dataset['chromSize']
            
            # Remove duplicates
            galaxy_dataset = galaxy_dataset.drop_duplicates(subset=['oSequence'])
            
            # Apply filters
            cleaned_dataset = galaxy_dataset[
                (galaxy_dataset['status'] == "Manually reviewed (Swiss-Prot)") &
                (galaxy_dataset['aa_length'] >= 50) &
                (galaxy_dataset['aa_length'] <= 10000)
            ]
            
            # Save individual file
            filename = f"uniprot_{species}.csv"
            cleaned_dataset.to_csv(filename, index=False)
            
            all_dfs.append(cleaned_dataset)
            print(f"  {len(cleaned_dataset)} proteins after filtering")
            
        except Exception as e:
            print(f"Error processing {species}: {e}")
    
    if not all_dfs:
        raise Exception("No data was processed successfully")
    
    # Combine all dataframes
    all_uniprot = pd.concat(all_dfs, ignore_index=True)
    all_uniprot.to_csv(combined_file, index=False)
    
    print(f"Combined dataset created with {len(all_uniprot)} proteins")
    return all_uniprot

# ============= ANALYSIS DEMO =============

# Example usage in a notebook
if __name__ == "__main__":
    # Load the data (replace with your own loading code if needed)
    all_uniprot = load_species_data()
    
    # Display basic info
    print(f"Total proteins: {len(all_uniprot)}")
    print(f"Species included: {', '.join(all_uniprot['species'].unique())}")
    
    # Filter for analysis
    min_length = 50
    max_length = 600
    filtered_proteins = all_uniprot[
        (all_uniprot['aa_length'] >= min_length) & 
        (all_uniprot['aa_length'] <= max_length)
    ]
    
    # Show protein count by species
    species_counts = filtered_proteins['species'].value_counts().reset_index()
    species_counts.columns = ['Species', 'Count']
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Species', y='Count', data=species_counts)
    plt.xticks(rotation=45)
    plt.title('Protein Count by Species (50-600 aa)')
    plt.tight_layout()
    
    # Show length distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(data=filtered_proteins, x='aa_length', bins=50)
    plt.title('Protein Length Distribution (50-600 aa)')
    plt.tight_layout()
    
    # Show species breakdown by pie chart
    plt.figure(figsize=(10, 10))
    filtered_proteins['species'].value_counts().plot.pie(autopct='%1.1f%%', textprops={'fontsize': 12})
    plt.title('Species Distribution in Dataset')
    plt.tight_layout()
