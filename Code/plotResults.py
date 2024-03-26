import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import poisson, chisquare, ks_2samp
from termcolor import colored

plt.rcParams.update({
    'font.size': 18,
    'font.family': 'serif',
})

def main(file):
    with open(file,'r') as f:
        # Read the results and plot the histograms of every team
        results = f.readlines()
        results = [x.strip() for x in results]
        results = [x.split(',') for x in results]
        results = results[1:]

        for team in results:
            name = team[0]
            # Convert string values to integers (except for the team name)
            counts = np.array([int(item) if item.isdigit() else item for item in team[1:]])

            # Calculate the total count of finishes
            total_count = counts.sum()

            # Calculate the probabilities for each position
            probabilities = counts / total_count
            # Get indices and probabilities where probability is over 0
            indices, probabilities_over_zero = zip(*[(i, prob) for i, prob in enumerate(probabilities, start=1) if prob > 0])
            probabilities_over_zero=list(probabilities_over_zero)
            indices=list(indices)
            print(f'Probability of Champions League for {name}: {probability_of_champions_league_place(probabilities):.2%}')
            print(f'Probability of relegation for {name}: {probability_of_relegation_place(probabilities):.2%}')
            
            # Calculate the average position
            positions = np.arange(1, len(counts) + 1)
            avg_position = (probabilities*positions).sum()

            # Fit a Poisson distribution to the data
            # Not necessary needed
            lambda_ = avg_position
            rv = poisson(lambda_)
            poisson_pmf = rv.pmf(range(1, len(counts) + 1))
            
           # Perform goodness-of-fit test
            '''chi2_stat, chi2_p_value = perform_goodness_of_fit_test(counts, poisson_pmf)
            if chi2_p_value > 0.05:  # Adjust the threshold as needed
                chi2_color = 'green'
            else:
                chi2_color = 'red'
            print(colored(f'Chi-square test for {name}: Statistic = {chi2_stat}, p-value = {chi2_p_value}', color=chi2_color))-'''

            # Perform Kolmogorov-Smirnov test
            '''ks_stat, ks_p_value = perform_kolmogorov_smirnov_test(counts, poisson_pmf)
            if ks_p_value > 0.05:  # Adjust the threshold as needed
                ks_color = 'green'
            else:
                ks_color = 'red'
            print(colored(f'Kolmogorov-Smirnov test for {name}: Statistic = {ks_stat}, p-value = {ks_p_value}', color=ks_color))'''

            plt.figure(figsize=(10, 6))
            plt.bar(indices, probabilities_over_zero, label='Position Probabilities')
            #plt.plot(range(1, len(counts) + 1), total_count * poisson_pmf, marker='o', linestyle='-', color='red', label='Poisson Distribution (λ = {:.2f})'.format(lambda_))
            plt.title(f'Results for {name}')
            plt.xlabel('Position')
            plt.ylabel('Probabilities')
            plt.xticks(indices)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            os.makedirs(os.path.dirname(f"DataPlotting/Figures/{name}/"), exist_ok=True)
            plt.savefig(f'DataPlotting/Figures/{name}/{name}_Results.pdf', format='pdf')
    
    return


def perform_goodness_of_fit_test(counts, poisson_pmf):
    # Calculate expected counts
    expected_counts = poisson_pmf * counts.sum()
    # Normalize expected counts so their sum equals the total count
    normalized_expected_counts = expected_counts / expected_counts.sum()
    # Normalize observed counts
    normalized_counts = counts / counts.sum()
    # Perform chi-square goodness-of-fit test
    chi2_stat, p_value = chisquare(normalized_counts, f_exp=normalized_expected_counts)
    return chi2_stat, p_value

def perform_kolmogorov_smirnov_test(counts, poisson_pmf):
    # Perform Kolmogorov-Smirnov test
    ks_stat, p_value = ks_2samp(counts, poisson_pmf * counts.sum())
    return ks_stat, p_value

def probability_of_relegation_place(probabilities):
    return sum(probabilities[-3:])

def probability_of_champions_league_place(probabilities):
    return sum(probabilities[:4])

main('Data\Results.csv')