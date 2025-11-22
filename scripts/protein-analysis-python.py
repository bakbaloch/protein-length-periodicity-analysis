import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import signal
from scipy import optimize
import seaborn as sns
import requests
from io import StringIO
import os
from tqdm import tqdm

# Define URLs and species names - same as in the R script
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

# Step 1: Function to consolidate UniProt data
def consolidate_uniprot(galaxy_url, species_name):
    """
    Downloads and preprocesses protein data from Galaxy URL for a specific species.
    
    Parameters:
    -----------
    galaxy_url : str
        URL to download the data
    species_name : str
        Name of the species
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned dataset
    """
    print(f"Processing {species_name}...")
    
    # Read the data
    try:
        response = requests.get(galaxy_url)
        response.raise_for_status()
        galaxy_dataset_all = pd.read_csv(StringIO(response.text))
    except Exception as e:
        print(f"Error reading data for {species_name}: {e}")
        return None
    
    # Apply transformations
    try:
        # Calculate amino acid length
        galaxy_dataset_all['species'] = species_name
        galaxy_dataset_all['aa_length'] = galaxy_dataset_all['oSequence'].str.len()
        
        # Calculate gene size and other metrics
        galaxy_dataset_all['gene_size'] = galaxy_dataset_all['chromEnd'] - galaxy_dataset_all['chromStart']
        galaxy_dataset_all['intergenic_distance'] = galaxy_dataset_all['chromStart'] - galaxy_dataset_all['chromStart'].shift(1)
        galaxy_dataset_all['relative_pos'] = galaxy_dataset_all['chromStart'] / galaxy_dataset_all['chromSize']
        
        # Remove duplicates
        galaxy_dataset_all = galaxy_dataset_all.drop_duplicates(subset=['oSequence'])
        
        # Apply filters
        cleaned_galaxy_dataset = galaxy_dataset_all[
            (galaxy_dataset_all['status'] == "Manually reviewed (Swiss-Prot)") &
            (galaxy_dataset_all['aa_length'] >= 50) &
            (galaxy_dataset_all['aa_length'] <= 10000)
        ]
        
        # Create dynamic filename
        filename = f"uniprot_{species_name}.csv"
        cleaned_galaxy_dataset.to_csv(filename, index=False)
        
        return cleaned_galaxy_dataset
    
    except Exception as e:
        print(f"Error processing data for {species_name}: {e}")
        return None

# Step 2: Process all species
def process_all_species():
    """Process and combine data from all species"""
    all_dfs = []
    
    for url, species in zip(galaxy_urls, species_list):
        df = consolidate_uniprot(url, species)
        if df is not None:
            all_dfs.append(df)
    
    if not all_dfs:
        print("No data was processed successfully.")
        return None
    
    # Combine all dataframes
    all_uniprot = pd.concat(all_dfs, ignore_index=True)
    all_uniprot.to_csv("uniprot_All_Species.csv", index=False)
    
    print(f"Combined dataset created with {len(all_uniprot)} proteins.")
    return all_uniprot

# Step 3: Generate species summary statistics
def generate_species_summary(all_uniprot, periodicity_results=None):
    """
    Generate summary statistics for each species
    
    Parameters:
    -----------
    all_uniprot : pandas.DataFrame
        Combined dataset of all species
    periodicity_results : dict, optional
        Results from periodicity analysis
        
    Returns:
    --------
    pandas.DataFrame
        Summary statistics by species
    """
    summary_list = []
    
    for species in all_uniprot['species'].unique():
        species_data = all_uniprot[all_uniprot['species'] == species]
        
        # Calculate basic statistics
        total_proteins = len(species_data)
        mean_length = species_data['aa_length'].mean()
        median_length = species_data['aa_length'].median()
        min_length = species_data['aa_length'].min()
        max_length = species_data['aa_length'].max()
        sd_length = species_data['aa_length'].std()
        
        # Count proteins in analysis range (50-600 aa)
        in_range = species_data[(species_data['aa_length'] >= 50) & (species_data['aa_length'] <= 600)]
        proteins_in_range = len(in_range)
        percent_in_range = (proteins_in_range / total_proteins) * 100 if total_proteins > 0 else 0
        
        # Calculate periodicity stats if available
        periodic_proteins = 0
        periodic_percent = 0
        
        if periodicity_results and 'enhanced_proteins' in periodicity_results:
            periodic_df = periodicity_results['enhanced_proteins']
            species_periodic = periodic_df[
                (periodic_df['species'] == species) & 
                (periodic_df['aa_length'] >= 50) & 
                (periodic_df['aa_length'] <= 600) & 
                (periodic_df['is_period_related'] == True)
            ]
            periodic_proteins = len(species_periodic)
            periodic_percent = (periodic_proteins / proteins_in_range) * 100 if proteins_in_range > 0 else 0
        
        # Create summary row
        summary_row = {
            'Species': species,
            'Total_Proteins': total_proteins,
            'Mean_Length': mean_length,
            'Median_Length': median_length,
            'Min_Length': min_length,
            'Max_Length': max_length,
            'SD_Length': sd_length,
            'Proteins_In_Range': proteins_in_range,
            'Percent_In_Range': percent_in_range,
            'Periodic_Proteins': periodic_proteins,
            'Periodic_Percent': periodic_percent
        }
        
        summary_list.append(summary_row)
    
    # Create DataFrame and sort
    summary_df = pd.DataFrame(summary_list)
    summary_df = summary_df.sort_values('Total_Proteins', ascending=False)
    
    # Save to CSV
    summary_df.to_csv("species_summary_table.csv", index=False)
    
    return summary_df

# Step 4: Prepare data for spectral analysis
def prepare_for_sad(data):
    """
    Prepare protein length data for spectral analysis
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Filtered protein data
        
    Returns:
    --------
    pandas.DataFrame
        Complete count data with zeros for missing lengths
    """
    # Create frequency table of protein lengths
    length_counts = data['aa_length'].value_counts().reset_index()
    length_counts.columns = ['aa_length', 'n']
    length_counts = length_counts.sort_values('aa_length')
    
    # Define complete range of lengths
    min_length = length_counts['aa_length'].min()
    max_length = length_counts['aa_length'].max()
    all_lengths = pd.DataFrame({'aa_length': range(min_length, max_length + 1)})
    
    # Join with counts and fill missing values with zero
    complete_counts = pd.merge(all_lengths, length_counts, on='aa_length', how='left')
    complete_counts['n'] = complete_counts['n'].fillna(0)
    
    return complete_counts

# Step 5: Spectral Analysis of Distributions (SAD)
def sad_analysis(total_vector, min_period=2, max_period=200):
    """
    Perform spectral analysis on protein length distribution
    
    Parameters:
    -----------
    total_vector : array-like
        Vector of protein counts by length
    min_period : int
        Minimum period to test
    max_period : int
        Maximum period to test
        
    Returns:
    --------
    pandas.DataFrame
        Results with period and amplitude
    """
    total_vector = np.array(total_vector)
    periods = range(min_period, max_period + 1)
    amplitudes = []
    
    for j in tqdm(periods, desc="Analyzing periods"):
        # Calculate non-oscillating background using moving average
        # In Python we can use convolution to implement the moving average
        window = np.ones(j) / j
        # Ensure the length is odd to match R's centered moving average
        if j % 2 == 0:
            nonoscj = np.convolve(total_vector, window, mode='valid')
            # Pad to match original length
            pad_width = j // 2
            nonoscj = np.pad(nonoscj, (pad_width, pad_width), 'edge')
        else:
            nonoscj = np.convolve(total_vector, window, mode='same')
        
        # Calculate oscillating component
        # Exclude the edges to avoid edge effects
        half_j = j // 2
        valid_indices = range(half_j, len(total_vector) - half_j)
        oscj = total_vector[valid_indices] - nonoscj[valid_indices]
        
        # Fit cosine function and extract amplitude
        t = np.arange(len(oscj))
        cos_component = np.cos(2 * np.pi * t / j)
        
        # Calculate amplitude using Fourier coefficient formula
        if np.sum(cos_component**2) > 0:
            amplitude = np.sum(oscj * cos_component) / np.sum(cos_component**2)
        else:
            amplitude = 0
            
        amplitudes.append(amplitude)
    
    return pd.DataFrame({'period': periods, 'amplitude': amplitudes})

# Step 6: Gamma PDF for mixture model
def gamma_pdf(x, alpha, beta):
    """
    Normalized gamma probability density function
    
    Parameters:
    -----------
    x : array-like
        Values at which to evaluate the PDF
    alpha : float
        Shape parameter
    beta : float
        Scale parameter
        
    Returns:
    --------
    array-like
        Normalized PDF values
    """
    raw_pdf = x**alpha * np.exp(-x/beta) / (scipy.special.gamma(alpha + 1) * beta**(alpha + 1))
    total = np.sum(raw_pdf)
    return raw_pdf / total if total > 0 else raw_pdf

# Step 7: Normal PDF for mixture model
def normal_pdf(x, mu, sigma):
    """
    Normalized normal probability density function
    
    Parameters:
    -----------
    x : array-like
        Values at which to evaluate the PDF
    mu : float
        Mean
    sigma : float
        Standard deviation
        
    Returns:
    --------
    array-like
        Normalized PDF values
    """
    raw_pdf = np.exp(-(x - mu)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    total = np.sum(raw_pdf)
    return raw_pdf / total if total > 0 else raw_pdf

# Step 8: Mixture Model Analysis
def mixture_model_analysis(length_data, preferred_period, k=4):
    """
    Implement mixture model analysis for protein length distribution
    
    Parameters:
    -----------
    length_data : pandas.DataFrame
        Data with aa_length and count columns
    preferred_period : float
        Preferred period from spectral analysis
    k : int
        Number of normal components to include
        
    Returns:
    --------
    dict
        Model results
    """
    from scipy import stats
    from scipy import optimize
    
    try:
        # Extract the range and prepare the data
        imin = length_data['aa_length'].min()
        imax = length_data['aa_length'].max()
        
        # Transform to vector format for modeling
        lengths = np.arange(imin, imax + 1)
        counts = np.zeros(len(lengths))
        
        for _, row in length_data.iterrows():
            idx = int(row['aa_length'] - imin)
            if 0 <= idx < len(counts):
                counts[idx] = row['count']
        
        # Total number of proteins
        n = np.sum(counts)
        
        # Negative log-likelihood function for the full model
        def full_model_nll(params):
            mu, sigma, alpha, beta, p1, p2, p3, p4 = params
            
            # Parameter constraints
            if (sigma <= 0 or beta <= 0 or alpha < 0 or
                p1 < 0 or p2 < 0 or p3 < 0 or p4 < 0 or
                (p1 + p2 + p3 + p4) >= 1):
                return 1e10  # Return very high value for invalid parameters
            
            # Calculate the background component (gamma distribution)
            g_pdf = gamma_pdf(lengths, alpha, beta)
            
            # Calculate normal distributions for each multiple of the period
            f1_pdf = normal_pdf(lengths, mu, sigma)
            f2_pdf = normal_pdf(lengths, 2*mu, np.sqrt(2)*sigma)
            f3_pdf = normal_pdf(lengths, 3*mu, np.sqrt(3)*sigma)
            f4_pdf = normal_pdf(lengths, 4*mu, np.sqrt(4)*sigma)
            
            # Calculate the mixture PDF
            mixture_pdf = (1 - p1 - p2 - p3 - p4) * g_pdf + \
                         p1 * f1_pdf + p2 * f2_pdf + p3 * f3_pdf + p4 * f4_pdf
            
            # Calculate the negative log-likelihood
            # Add small constant to avoid log(0)
            nll = -np.sum(counts * np.log(mixture_pdf + 1e-10))
            
            return nll
        
        # Negative log-likelihood function for the null model (background only)
        def background_only_nll(params):
            alpha, beta = params
            
            if beta <= 0 or alpha < 0:
                return 1e10
            
            g_pdf = gamma_pdf(lengths, alpha, beta)
            nll = -np.sum(counts * np.log(g_pdf + 1e-10))
            
            return nll
        
        # Initial parameter guesses
        initial_mu = preferred_period
        initial_sigma = preferred_period / 10
        
        # Use mean and variance to estimate initial gamma parameters
        mean_val = np.sum(lengths * counts) / np.sum(counts)
        var_val = np.sum(counts * (lengths - mean_val)**2) / np.sum(counts)
        
        initial_beta = var_val / mean_val
        initial_alpha = (mean_val / initial_beta) - 1
        
        # Initial peak probabilities
        initial_p1 = 0.01
        initial_p2 = 0.03
        initial_p3 = 0.07
        initial_p4 = 0.12
        
        # Fit the full model
        print("Fitting full model...")
        full_model_result = optimize.minimize(
            full_model_nll,
            x0=[initial_mu, initial_sigma, initial_alpha, initial_beta, initial_p1, initial_p2, initial_p3, initial_p4],
            method='L-BFGS-B',
            bounds=[(50, 200), (1, 50), (0, 10), (1, 1000), (0, 0.2), (0, 0.2), (0, 0.2), (0, 0.2)]
        )
        
        # Fit the null model
        print("Fitting null model...")
        null_model_result = optimize.minimize(
            background_only_nll,
            x0=[initial_alpha, initial_beta],
            method='L-BFGS-B',
            bounds=[(0, 10), (1, 1000)]
        )
        
        # Extract parameter estimates
        params = full_model_result.x
        mu_hat, sigma_hat, alpha_hat, beta_hat, p1_hat, p2_hat, p3_hat, p4_hat = params
        
        # Calculate the background parameters
        background_params = null_model_result.x
        alpha0_hat, beta0_hat = background_params
        
        # Convert to more intuitive parameters
        mu_background = beta_hat * (alpha_hat + 1)
        sigma_background = beta_hat * np.sqrt(alpha_hat + 1)
        
        mu_pure_background = beta0_hat * (alpha0_hat + 1)
        sigma_pure_background = beta0_hat * np.sqrt(alpha0_hat + 1)
        
        # Likelihood ratio test
        L0 = null_model_result.fun
        L1 = full_model_result.fun
        lambda_stat = 2 * (L0 - L1)
        
        # Degrees of freedom: k+2
        df = k + 2
        p_value = stats.chi2.sf(lambda_stat, df=df)
        
        # Generate model fits for visualization
        gamma_background = gamma_pdf(lengths, alpha_hat, beta_hat)
        pure_gamma_background = gamma_pdf(lengths, alpha0_hat, beta0_hat)
        
        # Calculate the full model probability density
        full_model_pdf = (1 - p1_hat - p2_hat - p3_hat - p4_hat) * gamma_background
        
        # Add peaks if they exist
        if p1_hat > 0: 
            full_model_pdf = full_model_pdf + p1_hat * normal_pdf(lengths, mu_hat, sigma_hat)
        if p2_hat > 0: 
            full_model_pdf = full_model_pdf + p2_hat * normal_pdf(lengths, 2*mu_hat, np.sqrt(2)*sigma_hat)
        if p3_hat > 0: 
            full_model_pdf = full_model_pdf + p3_hat * normal_pdf(lengths, 3*mu_hat, np.sqrt(3)*sigma_hat)
        if p4_hat > 0: 
            full_model_pdf = full_model_pdf + p4_hat * normal_pdf(lengths, 4*mu_hat, np.sqrt(4)*sigma_hat)
        
        # Scale the PDFs to match the counts
        estimated_counts = full_model_pdf * n
        background_only_counts = pure_gamma_background * n
        
        # Prepare data for plotting
        model_data = pd.DataFrame({
            'length': lengths,
            'observed': counts,
            'estimated': estimated_counts,
            'background': background_only_counts
        })
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        plt.plot(model_data['length'], model_data['observed'], 'k-', alpha=0.5, label='Observed')
        plt.plot(model_data['length'], model_data['estimated'], 'b-', linewidth=1, label='Model')
        plt.plot(model_data['length'], model_data['background'], 'r--', label='Background')
        plt.title(f"Estimated Probability Density of Protein Lengths\nPeriod = {mu_hat:.2f} aa (p-value = {p_value:.6f})")
        plt.xlabel("Protein Length")
        plt.ylabel("Probability Density")
        plt.legend()
        plt.savefig('protein_length_model.png')
        plt.close()
        
        # Return results
        return {
            'mu': mu_hat,
            'sigma': sigma_hat,
            'mu_background': mu_background,
            'sigma_background': sigma_background,
            'p1': p1_hat,
            'p2': p2_hat,
            'p3': p3_hat,
            'p4': p4_hat,
            'p_value': p_value,
            'model_data': model_data
        }
    
    except Exception as e:
        print(f"Error in mixture model fitting: {e}")
        return None

# Step 9: Analyze Protein Periodicity
def analyze_protein_periodicity(protein_df, period, sigma, k_max=4):
    """
    Analyze periodicity in protein lengths
    
    Parameters:
    -----------
    protein_df : pandas.DataFrame
        DataFrame with protein information
    period : float
        Fundamental period
    sigma : float
        Standard deviation for window width
    k_max : int
        Maximum multiple of period to check
        
    Returns:
    --------
    dict
        Periodicity analysis results
    """
    # Function to determine if a protein length is near a multiple of the period
    def is_near_period_multiple(length, period, sigma, k_max):
        for k in range(1, k_max + 1):
            expected_length = k * period
            # Use 1 sigma to define the window width
            if abs(length - expected_length) <= sigma:
                return k  # Return which multiple it's near
        return 0  # Not near any multiple
    
    # Apply the function to each protein
    protein_df = protein_df.copy()
    protein_df['period_multiple'] = protein_df['aa_length'].apply(
        lambda x: is_near_period_multiple(x, period, sigma, k_max)
    )
    protein_df['is_period_related'] = protein_df['period_multiple'] > 0
    
    # Create period category labels
    def get_period_category(mult):
        if mult == 1:
            return f"~{round(period)} aa"
        elif mult > 1:
            return f"~{mult}×{round(period)} aa"
        else:
            return "Non-periodic"
    
    protein_df['period_category'] = protein_df['period_multiple'].apply(get_period_category)
    
    # Summarize proteins by period relationship
    period_summary = protein_df.groupby(['period_multiple', 'period_category']).agg(
        count=('aa_length', 'count')
    ).reset_index()
    
    period_summary['percent'] = period_summary['count'] / len(protein_df) * 100
    period_summary = period_summary.sort_values('period_multiple')
    
    # Create distribution plots
    plt.figure(figsize=(12, 8))
    plt.hist(protein_df['aa_length'], bins=range(int(min(protein_df['aa_length'])), 
                                                int(max(protein_df['aa_length'])), 5),
             color='lightblue', edgecolor='black', alpha=0.5)
    
    # Add vertical lines for period multiples
    for i in range(1, k_max + 1):
        plt.axvline(x=i*period, linestyle='--', color='red')
        plt.text(i*period, 0, f"{i}×", va='bottom', ha='center', color='red')
    
    plt.title(f"Protein Length Distribution with Period Multiples\nPeriod = {period:.1f} amino acids (±{sigma:.1f})")
    plt.xlabel("Protein Length (aa)")
    plt.ylabel("Count")
    plt.savefig('protein_length_distribution.png')
    plt.close()
    
    # Create a stacked histogram
    plt.figure(figsize=(12, 8))
    
    # Use seaborn for better color palette
    sns.histplot(data=protein_df, x='aa_length', hue='period_category', bins=range(
        int(min(protein_df['aa_length'])), int(max(protein_df['aa_length'])), 5))
    
    plt.title("Protein Length Distribution by Period Category")
    plt.xlabel("Protein Length (aa)")
    plt.ylabel("Count")
    plt.savefig('protein_length_by_category.png')
    plt.close()
    
    # Create a summary report
    print("\n=== PERIODIC PROTEIN ANALYSIS SUMMARY ===")
    print(f"Total proteins analyzed: {len(protein_df)}")
    print(f"Proteins near period multiples: {sum(protein_df['is_period_related'])} "
          f"({sum(protein_df['is_period_related'])/len(protein_df)*100:.1f}%)")
    print(f"Fundamental period: {period:.1f} amino acids (±{sigma:.1f})")
    print("\nProteins by period multiple:")
    
    for i in range(1, k_max + 1):
        count = sum(protein_df['period_multiple'] == i)
        percent = count/len(protein_df)*100
        print(f"  Multiple {i}: {count} ({percent:.1f}%)")
    
    # Return the results
    return {
        'enhanced_proteins': protein_df,
        'period_summary': period_summary,
        'periodic_proteins': protein_df[protein_df['is_period_related']]
    }

# Step 10: Main function to run the entire analysis
def main():
    """Run the complete protein analysis pipeline"""
    print("Starting protein analysis pipeline")
    
    # Check if files exist locally first to avoid downloading again
    all_files_exist = all(os.path.exists(f"uniprot_{species}.csv") for species in species_list)
    combined_file_exists = os.path.exists("uniprot_All_Species.csv")
    
    if all_files_exist and combined_file_exists:
        print("All individual species files and combined file exist locally. Reading from disk...")
        # Read the combined file
        all_uniprot = pd.read_csv("uniprot_All_Species.csv")
    elif all_files_exist:
        print("All individual species files exist. Combining them...")
        # Read individual files and combine
        all_dfs = []
        for species in species_list:
            df = pd.read_csv(f"uniprot_{species}.csv")
            all_dfs.append(df)
        all_uniprot = pd.concat(all_dfs, ignore_index=True)
        all_uniprot.to_csv("uniprot_All_Species.csv", index=False)
    else:
        print("Downloading and processing species data...")
        all_uniprot = process_all_species()
        if all_uniprot is None:
            print("Error processing data. Exiting.")
            return
    
    print(f"Total proteins in dataset: {len(all_uniprot)}")
    
    # Step 2: Data Cleaning and Filtering
    # Clean column names if needed
    if '#"chrom"' in all_uniprot.columns:
        all_uniprot = all_uniprot.rename(columns={'#"chrom"': 'chrom'})
    
    # Filter proteins by length range for analysis
    min_length = 50
    max_length = 600
    filtered_proteins = all_uniprot[
        (all_uniprot['aa_length'] >= min_length) & 
        (all_uniprot['aa_length'] <= max_length)
    ]
    print(f"Filtered proteins for analysis: {len(filtered_proteins)}")
    
    # Step 3: Exploratory Data Visualization
    plt.figure(figsize=(12, 8))
    plt.hist(filtered_proteins['aa_length'], bins=range(min_length, max_length, 5), 
             color='lightblue', edgecolor='darkblue', alpha=0.7)
    plt.title("Distribution of Protein Lengths Across All Species")
    plt.xlabel("Protein Length (aa)")
    plt.ylabel("Number of Proteins")
    plt.savefig('overall_histogram.png')
    plt.close()
    
    # Step 4: Prepare Data for Spectral Analysis
    sad_prepared_data = prepare_for_sad(filtered_proteins)
    
    # Step 5: Run SAD Analysis
    min_period = 2
    max_period = 200
    sad_results = sad_analysis(sad_prepared_data['n'].values, min_period, max_period)
    
    # Find the period with maximum amplitude
    max_peak = sad_results.loc[sad_results['amplitude'].idxmax()]
    preferred_period = max_peak['period']
    
    # Visualize SAD results
    plt.figure(figsize=(12, 8))
    plt.plot(sad_results['period'], sad_results['amplitude'])
    plt.scatter(preferred_period, max_peak['amplitude'], color='red', s=100)
    plt.title(f"Cosine Spectrum of Protein Lengths Across All Species\nMaximum amplitude at period = {preferred_period} aa")
    plt.xlabel("Period (aa)")
    plt.ylabel("Amplitude")
    plt.savefig('sad_results.png')
    plt.close()
    
    # Step 6: Prepare for Mixture Model Analysis
    length_counts_for_model = filtered_proteins['aa_length'].value_counts().reset_index()
    length_counts_for_model.columns = ['aa_length', 'count']
    
    # Step 7: Run the Mixture Model Analysis
    model_results = mixture_model_analysis(length_counts_for_model, preferred_period)
    
    # Step 8: Run the Periodicity Analysis
    if model_results is not None:
        periodicity_results = analyze_protein_periodicity(
            filtered_proteins, 
            model_results['mu'], 
            model_results['sigma']
        )
    else:
        # Fallback to using the period from SAD analysis
        print("Using SAD analysis period as fallback...")
        periodicity_results = analyze_protein_periodicity(
            filtered_proteins, 
            preferred_period, 
            preferred_period / 10  # Using a rough estimate for sigma
        )
    
    # Step 9: Generate Species Summary
    summary_table = generate_species_summary(all_uniprot, periodicity_results)
    print("\nSpecies Summary Table:")
    print(summary_table.to_string())
    
    # Step 10: Generate Final Summary Table
    final_summary = pd.DataFrame({
        'Analysis': ['All Species Combined'],
        'TotalProteins': [len(filtered_proteins)],
        'FundamentalPeriod': [model_results['mu'] if model_results is not None else preferred_period],
        'StandardDeviation': [model_results['sigma'] if model_results is not None else preferred_period/10],
        'PeriodicProteins': [sum(periodicity_results['enhanced_proteins']['is_period_related'])],
        'PeriodicPercentage': [sum(periodicity_results['enhanced_proteins']['is_period_related']) / len(filtered_proteins) * 100]
    })
    
    print("\nFinal Analysis Summary:")
    print(final_summary.to_string(index=False))
    final_summary.to_csv('final_summary.csv', index=False)
    
    print("\nAnalysis complete. All results saved to disk.")

# Run the main function if this script is executed directly
if __name__ == "__main__":
    main()