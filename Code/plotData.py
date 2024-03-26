import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from read2022_23_PL_data import make_List_Of_Teams, exampleScoreData
from scipy.stats import poisson, chisquare
import numpy as np
from termcolor import colored
plt.rcParams.update({
    'font.size': 18,
    'font.family': 'serif',
})

matches_per_team = 38
home_games_per_team = matches_per_team / 2
away_games_per_team = matches_per_team / 2
matches_overall = 380
dropped_teams=["Leeds United", "Southampton", "Leicester City"]
promoted_teams = ["Burnley","Sheffield Utd","Luton Town"]

# Plot the data you want to see
def plotOptaData(team_list, example_team, season_xg):
    name=[]
    GF=[]
    xG=[]
    GA=[]
    xGA=[]
    example_xG=0
    example_xGA=0
    
    for team in team_list:
        name.append(team.squad)
        GF.append(team.GF)
        xG.append(team.xG) 
        GA.append(team.GA)
        xGA.append(team.xGA)
        
        if team.squad == example_team.name:
            example_xG = season_xg[team.squad]['xG'] / matches_per_team
            example_xGA = season_xg[team.squad]['xGA'] / matches_per_team
            print(team.squad, example_xG, example_xGA)
        
    exampleHis(example_team, example_xG)
    exampleHis2(example_team, example_xGA)
    
    plt.close()
    
    return

def plotF(x, y, y2):
    plt.figure(figsize=(12, 6))  
    plt.plot(x, y, label='GF', marker='o')
    plt.plot(x, y2, label=r'$\hat{xG}$', marker='o', linestyle='dashed', alpha=0.5)  
    plt.xlabel('Teams')
    plt.ylabel('Goals')
    plt.title('Goals and Expected Goals for Premier League Teams 2022-23')
    plt.xticks(rotation=45, ha='right') 
    plt.legend()
    plt.tight_layout()
    plt.grid('minor')
    plt.savefig("DataPlotting/Figures/Goals_F")
    
    return

def plotA(x, y, y2):
        
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, label='GA', marker='o')
    plt.plot(x, y2, label=r'$\hat{xGA}$', marker='o', linestyle='dashed' , alpha=0.5)  
    plt.xlabel('Teams')
    plt.ylabel('Goals')
    plt.title('Goals Against and Expected Goals Against for Premier League Teams 2022-23')
    plt.xticks(rotation=45, ha='right') 
    plt.legend()
    plt.tight_layout()
    plt.grid('minor')
    plt.savefig("DataPlotting/Figures/Goals_A")
    
    return

def exampleHis(example_team, xG):
    goals = example_team.homeGoals
    for goal in example_team.awayGoals:
        goals.append(goal)
        
    plt.figure(figsize=(12, 6))
    plt.hist(goals, bins=range(min(goals), max(goals)+1), align='left', rwidth=0.8, edgecolor='black')
    
    lambda_param = xG
    x_vals = np.arange(min(goals), max(goals) + 1)
    poisson_vals = poisson.pmf(x_vals, lambda_param) * len(goals)
    
    plt.plot(x_vals, poisson_vals, 'r-', label=f'Poisson Distribution (λ={lambda_param:.2f})')
    
    plt.xlabel('Goals')
    plt.ylabel('Frequency')
    plt.title(f'Goal Distribution for {example_team.name} in Premier League 2022-23')
    plt.grid(axis='y', alpha=0.75)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(f"DataPlotting/Figures/{example_team.name}/"), exist_ok=True)
    plt.savefig(f"DataPlotting/Figures/{example_team.name}/Example_Histogram_{example_team.name}.pdf", format="pdf")
    
    observed_counts, _ = np.histogram(goals, bins=range(min(goals), max(goals) + 2))
    
    observed_counts_normalized = observed_counts / np.sum(observed_counts)
    poisson_vals_normalized = poisson_vals / np.sum(poisson_vals)

    chi2, p_value = chisquare(observed_counts_normalized, poisson_vals_normalized)
    if p_value < 0.05:
        color = 'red'
    else:
        color = 'green'

    print(colored(f"Chi-square test statistic for Goals Forward: {round(chi2,3)}", color))
    print(colored(f"P-value for Goals Forward: {round(p_value,3)}", color))
    
    plt.close()
    
    return

def exampleHis2(example_team, xGA):
    goals_against = example_team.homeGoalsAgainst
    for goal in example_team.awayGoalsAgainst:
        goals_against.append(goal)
        
    plt.figure(figsize=(12, 6))
    plt.hist(goals_against, bins=range(min(goals_against), max(goals_against)+1), align='left', rwidth=0.8, edgecolor='black')
    
    lambda_param = xGA
    x_vals = np.arange(min(goals_against), max(goals_against) + 1)
    poisson_vals = poisson.pmf(x_vals, lambda_param) * len(goals_against)
    
    plt.plot(x_vals, poisson_vals, 'r-', label=f'Poisson Distribution (λ={lambda_param:.2f})')
    
    plt.xlabel('Goals Against')
    plt.ylabel('Frequency')
    plt.title(f'Goals Against Distribution for {example_team.name} in Premier League 2022-23')
    plt.grid(axis='y', alpha=0.75)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(f"DataPlotting/Figures/{example_team.name}/"), exist_ok=True)
    plt.savefig(f"DataPlotting/Figures/{example_team.name}/Example_Histogram_{example_team.name}_Against.pdf", format="pdf")
    
    observed_counts, _ = np.histogram(goals_against, bins=range(min(goals_against), max(goals_against) + 2))

    observed_counts_normalized = observed_counts / np.sum(observed_counts)
    poisson_vals_normalized = poisson_vals / np.sum(poisson_vals)

    chi2, p_value = chisquare(observed_counts_normalized, poisson_vals_normalized)
    
    if p_value < 0.05:
        color = 'red'
    else:
        color = 'green'

    print(colored(f"Chi-square test statistic for Goals Against: {round(chi2,3)}", color))
    print(colored(f"P-value for Goals Against: {round(p_value,3)}\n", color))
    
    plt.close()
    
    return

def plotErrorHis(team_list):
    name=[]
    GF_Error=[]
    GA_Error=[]
    
    for team in team_list:
        name.append(team.squad)
        GF_Error.append((team.GF - team.xG) / matches_per_team)
        GA_Error.append((team.GA - team.xGA) / matches_per_team)
        
    plotGF_Error(GF_Error)
    plotGA_Error(GA_Error)
    
    return

def plotGF_Error(GF_Error):
    
    plt.figure(figsize=(12, 6))
    plt.hist(GF_Error, bins=15, color='blue', alpha=0.7, label='GF Error')
    plt.xlabel('GF Error (Actual Goals - Expected Goals)')
    plt.ylabel('Frequency')
    plt.title('Histogram of GF Error for Premier League Teams 2022-23')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig("DataPlotting/Figures/GF_Error_Histogram")
    
    return

def plotGA_Error(GA_Error):
    
    plt.figure(figsize=(12, 6))
    plt.hist(GA_Error, bins=15, color='red', alpha=0.7, label='GA Error')
    plt.xlabel('GA Error (Actual Goals Conceded - Expected Goals Conceded)')
    plt.ylabel('Frequency')
    plt.title('Histogram of GA Error for Premier League Teams 2022-23')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig("DataPlotting/Figures/GA_Error_Histogram")
    
    return

def getTeamSeasonxG(home_team, away_team, advantages):
    aha, dha, aaa, daa = advantages
    
    home_goals = sum(home_team.homeGoals)
    home_goals_against = sum(home_team.homeGoalsAgainst)
    away_goals = sum(away_team.awayGoals)
    away_goals_against = sum(away_team.awayGoalsAgainst)
    
    has = (home_goals / home_games_per_team) / aha
    hds = (home_goals_against / home_games_per_team) / dha
    
    aas = (away_goals / away_games_per_team) / aaa
    ads = (away_goals_against / away_games_per_team) / daa

    home_xG = has * ads * aha
    away_xG = aas * hds * aaa
    
    season_xG[home_team.name]['xG'] += home_xG
    season_xG[home_team.name]['xGA'] += away_xG 
    season_xG[away_team.name]['xG'] += away_xG
    season_xG[away_team.name]['xGA'] += home_xG
    
    return

advantages=[1.6162165371315789, 1.1776315789473684, 1.1649942485526317, 1.5865131578947365]

season_xG = {team.squad: {'xG': 0, 'xGA': 0} for team in make_List_Of_Teams("Data\Table_2022-23.csv")}
with open("Data/matches.txt","r") as f:
        for line in f:
            team1 = line[:-1].split("-")[0].split("|")[0][:-1]
            team2 = line[:-1].split("-")[1].split("|")[1][1:]
            if team1 == promoted_teams[0]:
                team1 = dropped_teams[0]
            if team2 == promoted_teams[0]:
                team2 = dropped_teams[0]
            if team1 == promoted_teams[1]:
                team1 = dropped_teams[1]
            if team2 == promoted_teams[1]:
                team2 = dropped_teams[1]
            if team1 == promoted_teams[2]:
                team1 = dropped_teams[2]
            if team2 == promoted_teams[2]:
                team2 = dropped_teams[2]
            home_xg = float(line[:-1].split("-")[0].split("|")[1])
            away_xg = float(line[:-1].split("-")[1].split("|")[0])
            getTeamSeasonxG(exampleScoreData("Data\Scores_2022-23.csv",team1), exampleScoreData("Data\Scores_2022-23.csv",team2), advantages)
f.close()
            
for team in make_List_Of_Teams("Data\Table_2022-23.csv"):
    plotOptaData(make_List_Of_Teams("Data\Table_2022-23.csv"), exampleScoreData("Data\Scores_2022-23.csv",team.squad), season_xG)
plotErrorHis(make_List_Of_Teams("Data\Table_2022-23.csv"))
