import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy import stats
from sklearn import linear_model
import math
plt.rcParams.update({'font.size': 18})

# Could be used if wanted
def multivariable_linear_regression(x_values, y_values):
    x_values = np.array(x_values)
    y_values = np.array(y_values)

    regr = linear_model.LinearRegression()
    regr.fit(x_values, y_values)

    print(f"Linear regression coefficients: {regr.coef_[0]}")
    print(f"Intercept: {regr.intercept_}")
    print(f"R-squared: {regr.score(x_values, y_values)}")
    
    predicted_y = regr.predict(x_values)
    residuals = y_values - predicted_y
    plt.scatter(predicted_y, residuals)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Analysis')
    plt.axhline(y=0, color='r', linestyle='--')  # Horizontal line at y=0
    plt.show()
    
    return regr.coef_[0], regr.intercept_


def linear_regression_and_mse(x_values, y_values, title, filename, xlabel, ylabel):
    plt.figure(figsize=(12, 8))
    plt.plot(x_values, y_values, label=f'{title}', marker='o', linestyle="")
    
    # Fit a linear regression line
    #z = np.polyfit(x_values, y_values, 1)
    #p = np.poly1d(z)
    result=stats.linregress(x_values,np.array(y_values))
    predicted_y = result.intercept + result.slope * np.array(x_values)
    plt.plot(x_values, predicted_y, "--", label=f"Linear Regression {round(result.slope,2)}x + {round(result.intercept,2)}")

    plt.xlabel('Championship')
    plt.ylabel('Premier')
    plt.title(title)
    plt.xticks(rotation=45, ha='right') 
    plt.legend()
    plt.tight_layout()
    plt.grid('minor')
    plt.savefig(f"DataPlotting/Figures/Promoted/{filename}")

    print(f"{title}: R-squared: {round(result.rvalue**2,3)}, P-value: {round(result.pvalue,3)}")
    #mse = mean_squared_error(y_values, p(x_values))
    #print(f"MSE for {title}: {round(mse, 2)}")
    
    return [result.slope,result.intercept]

# Doesn't really affect the results
def remove_outliers(data1, data2):
    z_scores1 = np.abs(stats.zscore(data1))
    z_scores2 = np.abs(stats.zscore(data2))
    
    threshold = 2
    
    # Identify outliers for each dataset
    outliers1 = z_scores1 > threshold
    outliers2 = z_scores2 > threshold

    # Combine the outlier masks
    combined_outliers = outliers1 | outliers2
    print(data1[combined_outliers],data2[combined_outliers])

    # Remove outliers from both datasets
    data1_cleaned = data1[~combined_outliers]
    data2_cleaned = data2[~combined_outliers]

    return data1_cleaned, data2_cleaned

def main(file):
    cHomeGoals=[]
    cHomeGoalsAgainst=[]
    cAwayGoals=[]
    cAwayGoalsAgainst=[]
    pHomeGoals=[]
    pHomeGoalsAgainst=[]
    pAwayGoals=[]
    pAwayGoalsAgainst=[]
    teams=[]
    sub1=[]
    sub2=[]
    p=[]
    
    with open(file,"r") as f:
        
        f.readline()
        
        for line in f:
            split_line=line[:-2].split("|")
            cHomeGoals.append(int(split_line[1].split("-")[0]))
            cHomeGoalsAgainst.append(int(split_line[1].split("-")[1]))
            cAwayGoals.append(int(split_line[2].split("-")[0]))
            cAwayGoalsAgainst.append(int(split_line[2].split("-")[1]))
            pHomeGoals.append(int(split_line[3].split("-")[0]))
            pHomeGoalsAgainst.append(int(split_line[3].split("-")[1]))
            pAwayGoals.append(int(split_line[4].split("-")[0]))
            pAwayGoalsAgainst.append(int(split_line[4].split("-")[1]))
            teams.append(split_line[0])
            #sub1.append(int(split_line[2].split("-")[0])-int(split_line[1].split("-")[0]))
            #sub2.append(int(split_line[2].split("-")[1])-int(split_line[1].split("-")[1]))
            
            
    #pHomeGoals, cHomeGoals = remove_outliers(np.array(pHomeGoals),np.array(cHomeGoals))
    #pHomeGoalsAgainst, cHomeGoalsAgainst = remove_outliers(np.array(pHomeGoalsAgainst),np.array(cHomeGoalsAgainst))
    #pAwayGoals, cAwayGoals = remove_outliers(np.array(pAwayGoals),np.array(cAwayGoals))
    #pAwayGoalsAgainst, cAwayGoalsAgainst = remove_outliers(np.array(pAwayGoalsAgainst),np.array(cAwayGoalsAgainst))
    p.append(linear_regression_and_mse(cHomeGoals, pHomeGoals, 'Linear Regression for Home Goals Forward in the Premier League', 'Home_GoalsF_LR','Championship Home Goals','Premier League Home Goals'))
    p.append(linear_regression_and_mse(cHomeGoalsAgainst, pHomeGoalsAgainst, 'Linear Regression for Home Goals Against in the Premier League', 'Home_GoalsA_LR','Championship Home Goals Against','Premier League Home Goals Against'))
    p.append(linear_regression_and_mse(cAwayGoals, pAwayGoals, 'Linear Regression for Away Goals Forward in the Premier League', 'Away_GoalsF_LR','Championship Away Goals','Premier League Away Goals'))
    p.append(linear_regression_and_mse(cAwayGoalsAgainst, pAwayGoalsAgainst, 'Linear Regression for Away Goals Against in the Premier League', 'Away_GoalsA_LR','Championship Goals Away Against','Premier League Goals Away Against'))
    
    #test=[]
    #for i in range(0,len(cHomeGoalsAgainst)):
        #test.append([cHomeGoalsAgainst[i],cAwayGoalsAgainst[i]])
    
    #print(multivariable_linear_regression(test,pHomeGoalsAgainst))   
       
    print(f"Home Goals Against Mean: {np.mean(pHomeGoalsAgainst)}\nAway Goals Against Mean: {np.mean(pAwayGoalsAgainst)}")     
         
    plt.figure(figsize=(12, 8))  
    plt.plot(teams, cHomeGoals, label='Home Goals in CS', marker='o')
    plt.plot(teams, pHomeGoals, label='Home Goals in PL', marker='o', linestyle='dashed', alpha=0.5)  
    plt.xlabel('Teams')
    plt.ylabel('Goals')
    plt.title('Home Goals in Premier League and Championship')
    plt.xticks(rotation=45, ha='right') 
    plt.legend()
    plt.tight_layout()
    plt.grid('minor')
    plt.savefig("DataPlotting/Figures/Promoted/Home_GoalsF")       
    
    plt.figure(figsize=(12, 8))  
    plt.plot(teams, cHomeGoalsAgainst, label='Home Goals Against in CS', marker='o')
    plt.plot(teams, pHomeGoalsAgainst, label='Home Goals Against in PL', marker='o', linestyle='dashed', alpha=0.5)  
    plt.xlabel('Teams')
    plt.ylabel('Goals')
    plt.title('Home Goals Against in Premier League and Championship')
    plt.xticks(rotation=45, ha='right') 
    plt.legend()
    plt.tight_layout()
    plt.grid('minor')
    plt.savefig("DataPlotting/Figures/Promoted/Home_GoalsA")
    
    plt.figure(figsize=(12, 8))  
    plt.plot(teams, cAwayGoals, label='Away Goals in CS', marker='o')
    plt.plot(teams, pAwayGoals, label='Away Goals in PL', marker='o', linestyle='dashed', alpha=0.5)  
    plt.xlabel('Teams')
    plt.ylabel('Goals')
    plt.title('Away Goals in Premier League and Championship')
    plt.xticks(rotation=45, ha='right') 
    plt.legend()
    plt.tight_layout()
    plt.grid('minor')
    plt.savefig("DataPlotting/Figures/Promoted/Away_GoalsF")       
    
    plt.figure(figsize=(12, 8))  
    plt.plot(teams, cAwayGoalsAgainst, label='Away Goals Against in CS', marker='o')
    plt.plot(teams, pAwayGoalsAgainst, label='Away Goals Against in PL', marker='o', linestyle='dashed', alpha=0.5)  
    plt.xlabel('Teams')
    plt.ylabel('Goals')
    plt.title('Away Goals Against in Premier League and Championship')
    plt.xticks(rotation=45, ha='right') 
    plt.legend()
    plt.tight_layout()
    plt.grid('minor')
    plt.savefig("DataPlotting/Figures/Promoted/Away_GoalsA")
        
    f.close()
    
    return p

print(main("Data\promoted_team_data.txt"))

# Linear regression for Home Goals Forward, where x is Goals in Championship: 0.47940413x + 0.0926532
# Linear regression for Home Goals Against: -1.69270833e-02x + 2.94110243e+01, Mean: 28.5
# Linear regression for Away Goals Forward: 0.24283789x + 8.28080145
# Linear regression for Away Goals Against: 0.17917775x + 28.67498466, Mean: 33.291666666666664