import time
import copy
import pandas as pd
import matplotlib.pyplot as plt
from read2022_23_PL_data import make_List_Of_Teams, exampleScoreData
from scipy.stats import poisson
import numpy as np

matches_per_team = 38
home_games_per_team = matches_per_team / 2
away_games_per_team = matches_per_team / 2
matches_overall = 380

promoted_teams = ["Burnley","Sheffield Utd","Luton Town"]
dropped_teams=["Leeds United", "Southampton", "Leicester City"]

def simulate_matches(home_xG, away_xG, num_simulations):
    # Get the match result samples from the Poisson distribution for the home and away teams
    home_goals = poisson.rvs(mu=home_xG, size=num_simulations)
    away_goals = poisson.rvs(mu=away_xG, size=num_simulations)
    
    #print(f"Home xG: {home_xG}, Home goals: {home_goals}, Away xG: {away_xG}, Away goals: {away_goals}")

    return home_goals, away_goals

def display_table():
    # Iterate through each season's results and update the position counts
    for team, positions in all_season_results.items():
        for pos in positions:
            position_counts[team][pos] += 1

    # Convert the dictionary to a Pandas DataFrame for easy tabular display
    df = pd.DataFrame(position_counts).transpose()

    # Display the DataFrame
    df.to_csv("Data/Results.csv")
    return
# Calculate advantages
'''home_goals=0
away_goals=0
home_goals_against=0
away_goals_against=0
for team in make_List_Of_Teams("Data\Table_2022-23.csv"):
    if team.squad in dropped_teams:
        home_goals+=exampleScoreData("Data\Scores_2022-23.csv",team.squad).homeGoals
        away_goals+=exampleScoreData("Data\Scores_2022-23.csv",team.squad).awayGoals
        home_goals_against+=exampleScoreData("Data\Scores_2022-23.csv",team.squad).homeGoalsAgainst
        away_goals_against+=exampleScoreData("Data\Scores_2022-23.csv",team.squad).awayGoalsAgainst
    else:
        home_goals+=sum(exampleScoreData("Data\Scores_2022-23.csv",team.squad).homeGoals)
        away_goals+=sum(exampleScoreData("Data\Scores_2022-23.csv",team.squad).awayGoals)
        home_goals_against+=sum(exampleScoreData("Data\Scores_2022-23.csv",team.squad).homeGoalsAgainst)
        away_goals_against+=sum(exampleScoreData("Data\Scores_2022-23.csv",team.squad).awayGoalsAgainst)
        
advantages=[home_goals/matches_overall,home_goals_against/matches_overall,away_goals/matches_overall,away_goals_against/matches_overall]'''

advantages=[1.6162165371315789, 1.1776315789473684, 1.1649942485526317, 1.5865131578947365]

# Start the simulation
number_of_sims=int(input("Enter the number of simulations: "))
start_time=time.time()

team_points = {team.squad: 0 for team in make_List_Of_Teams("Data\Table_2022-23.csv")}
team_goals_scored = {team.squad: 0 for team in make_List_Of_Teams("Data\Table_2022-23.csv")}
team_goals_conceded = {team.squad: 0 for team in make_List_Of_Teams("Data\Table_2022-23.csv")}
team_positions = {team.squad: 0 for team in make_List_Of_Teams("Data\Table_2022-23.csv")}
matchups = []
all_results = {f'match{i}': [] for i in range(1, matches_overall + 1)}
team_positions[promoted_teams[0]]=team_positions.pop(dropped_teams[0])
team_positions[promoted_teams[1]]=team_positions.pop(dropped_teams[1])
team_positions[promoted_teams[2]]=team_positions.pop(dropped_teams[2])
team_points[promoted_teams[0]]=team_points.pop(dropped_teams[0])
team_points[promoted_teams[1]]=team_points.pop(dropped_teams[1])
team_points[promoted_teams[2]]=team_points.pop(dropped_teams[2])
team_goals_scored[promoted_teams[0]]=team_goals_scored.pop(dropped_teams[0])
team_goals_scored[promoted_teams[1]]=team_goals_scored.pop(dropped_teams[1])
team_goals_scored[promoted_teams[2]]=team_goals_scored.pop(dropped_teams[2])
team_goals_conceded[promoted_teams[0]]=team_goals_conceded.pop(dropped_teams[0])
team_goals_conceded[promoted_teams[1]]=team_goals_conceded.pop(dropped_teams[1])
team_goals_conceded[promoted_teams[2]]=team_goals_conceded.pop(dropped_teams[2])
# Simulate the match results
with open("Data/matches.txt","r") as f:
    n=1
    for line in f:
        team1=line[:-1].split("-")[0].split("|")[0][:-1]
        home_xG=float(line[:-1].split("-")[0].split("|")[1][:-1])
        team2=line[:-1].split("-")[1].split("|")[1][1:]
        away_xG=float(line[:-1].split("-")[1].split("|")[0][:-1])
        matchups.append([team1, team2])
        home_goals, away_goals=simulate_matches(home_xG,away_xG,number_of_sims)
        all_results[f'match{n}'] = [home_goals, away_goals]
        n+=1
    n-=1

# To find the match results, from the n matchup use "all_results[f'match{n}']" where n is the matchup number
# If you want to find the home goals of season m use "all_results[f'match{n}'][0][m] and away goals all_results[f'match{n}'][1][m]"
for m in range(number_of_sims):
    if (m+1)%100==0:
        print(f"Simulating season {m+1}...")
    team_points = {team.squad: 0 for team in make_List_Of_Teams("Data\Table_2022-23.csv")}
    team_goals_scored = {team.squad: 0 for team in make_List_Of_Teams("Data\Table_2022-23.csv")}
    team_goals_conceded = {team.squad: 0 for team in make_List_Of_Teams("Data\Table_2022-23.csv")}
    team_positions = {team.squad: 0 for team in make_List_Of_Teams("Data\Table_2022-23.csv")}
    team_positions[promoted_teams[0]]=team_positions.pop(dropped_teams[0])
    team_positions[promoted_teams[1]]=team_positions.pop(dropped_teams[1])
    team_positions[promoted_teams[2]]=team_positions.pop(dropped_teams[2])
    team_points[promoted_teams[0]]=team_points.pop(dropped_teams[0])
    team_points[promoted_teams[1]]=team_points.pop(dropped_teams[1])
    team_points[promoted_teams[2]]=team_points.pop(dropped_teams[2])
    team_goals_scored[promoted_teams[0]]=team_goals_scored.pop(dropped_teams[0])
    team_goals_scored[promoted_teams[1]]=team_goals_scored.pop(dropped_teams[1])
    team_goals_scored[promoted_teams[2]]=team_goals_scored.pop(dropped_teams[2])
    team_goals_conceded[promoted_teams[0]]=team_goals_conceded.pop(dropped_teams[0])
    team_goals_conceded[promoted_teams[1]]=team_goals_conceded.pop(dropped_teams[1])
    team_goals_conceded[promoted_teams[2]]=team_goals_conceded.pop(dropped_teams[2])
    for k in range(1, matches_overall + 1):
        home_team=matchups[k-1][0]
        away_team=matchups[k-1][1]
        home_goal = all_results[f'match{k}'][0][m]
        away_goal = all_results[f'match{k}'][1][m]

        if home_goal > away_goal:
            team_points[home_team] += 3
        elif home_goal == away_goal:
            team_points[home_team] += 1
            team_points[away_team] += 1
        else:
            team_points[away_team] += 3

        team_goals_scored[home_team] += home_goal
        team_goals_conceded[home_team] += away_goal

        team_goals_scored[away_team] += away_goal
        team_goals_conceded[away_team] += home_goal
    
    # Store the results for this season
    season_results = {
        'team_points': copy.deepcopy(team_points),
        'team_goals_scored': copy.deepcopy(team_goals_scored),
        'team_goals_conceded': copy.deepcopy(team_goals_conceded)
    }

    # Calculate team positions for this season
    sorted_table = sorted(
        team_points.items(),
        key=lambda x: (x[1], team_goals_scored[x[0]], -team_goals_conceded[x[0]]),
        reverse=True,
    )
    prev_points = None
    prev_goal_difference = None
    prev_goals_forward = None
    current_position = 1

    for idx, (team, _) in enumerate(sorted_table, start=1):
        team_positions[team] = idx

    # Store the positions for this season
    season_positions = {team: [position] for team, position in team_positions.items()}
    if m==0:
        all_season_results = season_positions
    else:
        for team, position in team_positions.items():
            all_season_results[team].append(season_positions[team][0])

end_time=time.time()
elapsed_time = end_time - start_time
elapsed_time_minutes = elapsed_time / 60
remaining_seconds = elapsed_time % 60
print(f"Total time taken: {elapsed_time_minutes:.0f} minutes and {remaining_seconds:.2f} seconds")

# Initialize a dictionary to store the count of each position for each team
position_counts = {team: {pos: 0 for pos in range(1, 21)} for team in all_season_results.keys()}
display_table()


# Make schedule
'''
def makeSchedule(home_team, away_team, advantages):
    aha, dha, aaa, daa = advantages

    if home_team.name in promoted_teams:
        homeGoals=home_team.homeGoals
        homeGoalsAgainst=home_team.homeGoalsAgainst
    else:
        homeGoals=sum(home_team.homeGoals)
        homeGoalsAgainst=sum(home_team.homeGoalsAgainst)
        
    if away_team.name in promoted_teams:
        awayGoals=away_team.awayGoals
        awayGoalsAgainst=away_team.awayGoalsAgainst
    else:
        awayGoals=sum(away_team.awayGoals)
        awayGoalsAgainst=sum(away_team.awayGoalsAgainst)
    
    has=(homeGoals/home_games_per_team)/aha
    hds=(homeGoalsAgainst/home_games_per_team)/dha
    
    aas=(awayGoals/away_games_per_team)/aaa
    ads=(awayGoalsAgainst/away_games_per_team)/daa

    home_xG=has*ads*aha
    away_xG=aas*hds*aaa
    
    return home_xG, away_xG

file=open("Data/matches.txt","w")
for team in make_List_Of_Teams("Data\Table_2022-23.csv"):
    for team2 in make_List_Of_Teams("Data\Table_2022-23.csv"):
        if team.squad!=team2.squad:
            home=team.squad
            away=team2.squad
            if team.squad==dropped_teams[0]:
                home=promoted_teams[0]
            if team2.squad==dropped_teams[0]:
                away=promoted_teams[0]
            if team.squad==dropped_teams[1]:
                home=promoted_teams[1]
            if team2.squad==dropped_teams[1]:
                away=promoted_teams[1]
            if team.squad==dropped_teams[2]:
                home=promoted_teams[2]
            if team2.squad==dropped_teams[2]:
                away=promoted_teams[2] 
            home_xg, away_xg = makeSchedule(exampleScoreData("Data\Scores_2022-23.csv",team.squad),exampleScoreData("Data\Scores_2022-23.csv",team2.squad),advantages)
            file.write(f"{home} | {home_xg} - {away_xg} | {away}\n")
file.close()'''

# Linear regression for Home Goals Forward, where x is Goals in Championship: 0.47940413x + 0.0926532
# Linear regression for Home Goals Against: -1.69270833e-02x + 2.94110243e+01, Mean: 28.5
# Linear regression for Away Goals Forward: 0.24283789x + 8.28080145
# Linear regression for Away Goals Against: 0.17917775x + 28.67498466, Mean: 33.291666666666664