import pandas as pd

class Team:
    def __init__(self, rank, squad, GF, GA, GD, Pts, xG, xGA):
        self.rank = rank
        self.squad = squad
        self.GF = GF
        self.GA = GA
        self.GD = GD
        self.Pts = Pts
        self.xG = xG
        self.xGA = xGA
        
class Example:
    def __init__(self, name,  homeGoals, awayGoals, homeGoalsAgainst, awayGoalsAgainst):
        self.name = name
        self.homeGoals = homeGoals
        self.awayGoals = awayGoals
        self.homeGoalsAgainst = homeGoalsAgainst
        self.awayGoalsAgainst = awayGoalsAgainst
        
dropped_teams=[]    # When you want don't want to replace the relgated teams use this list
# dropped_teams=["Leeds United", "Southampton", "Leicester City"]   # When making the match schedule, these teams were replaced

def make_List_Of_Teams(file_name):
    team_list = []
    file = pd.read_csv(file_name, usecols=["Rk","Squad","GF","GA","GD","Pts","xG","xGA"])
    for i in range(0,len(file)):
        team = Team(file.iloc[i]["Rk"],file.iloc[i]["Squad"],file.iloc[i]["GF"],file.iloc[i]["GA"],file.iloc[i]["GD"],file.iloc[i]["Pts"],file.iloc[i]["xG"],file.iloc[i]["xGA"])
        team_list.append(team)
    return team_list

def exampleScoreData(file_name, team_name):
    home_goals=[]
    away_goals=[]
    home_goals_against=[]
    away_goals_against=[]
    file = pd.read_csv(file_name, delimiter=";", encoding="utf-8")
    file2 = open("Data\promoted_teams2023.txt","r")
    
    # Calculate the promoted teams' goals and goals against
    if team_name in dropped_teams:
        for i in range(0,dropped_teams.index(team_name)+1):
            line=file2.readline().split("|")
            team_name=line[0]
            home_goals=int(line[1].split("-")[0])*0.47940413+0.0926532
            home_goals_against=28.5
            away_goals=int(line[2].split("-")[0])*0.24283789+8.28080145
            away_goals_against=33.291666666666664
    else:    
        for i in range(0,len(file)):
            if (file.iloc[i]["Home"]==team_name):
                score = file.iloc[i]["Score"].split("–")
                home_goals.append(int(score[0]))
                home_goals_against.append(int(score[1]))
                
                
            if (file.iloc[i]["Away"]==team_name):
                score = file.iloc[i]["Score"].split("–")
                away_goals.append(int(score[1]))
                away_goals_against.append(int(score[0]))
        
    example=Example(team_name, home_goals, away_goals, home_goals_against, away_goals_against)
    
    file2.close()
            
    
    return example

# Test runs
#exampleScoreData("Data\Scores_2022-23.csv","Arsenal")
#make_List_Of_Teams("Data\Table_2022-23.csv")
#exampleScoreData("Data\Table_2022-23.csv","Leicester City").awayGoals

# Linear regression for Home Goals Forward, where x is Goals in Championship: 0.47940413x + 0.0926532
# Linear regression for Home Goals Against: -1.69270833e-02x + 2.94110243e+01, Mean: 28.5
# Linear regression for Away Goals Forward: 0.24283789x + 8.28080145
# Linear regression for Away Goals Against: 0.17917775x + 28.67498466, Mean: 33.291666666666664