import pandas as pd
import numpy as np

# webscrapping the table of the results and put it in the race_results df
url = 'https://www.formula1.com/en/results.html/2021/races.html'
table= pd.read_html(url)   
df=table[0]
df = df.drop(df.columns[0],axis=1)
race_results = df.drop(df.columns[-1],axis=1)

#keep the unique drivers only
drivers=race_results['Winner'].unique().tolist()

#keep the unique teams only
teams=race_results['Car'].unique().tolist()

#the indices of each driver
driver_indices = {driver: index for index, driver in enumerate(drivers)}

#the indices of each team
teams_indices = {team: index for index, team in enumerate(teams)}


#initialize the adjacency_matrix of the drivers with zeros
adjacency_matrix_drivers = np.zeros((len(drivers), len(drivers)))


#initialize the adjacency_matrix of the teams with zeros
adjacency_matrix_teams = np.zeros((len(teams), len(teams)))

#update the adjacency matrixes both for teams and for the drivers based on race results
for index, row in race_results.iterrows():
    driver1 = row['Winner']
    team1=row['Car']
    for i in driver_indices:
        if i==driver1:
            position=driver_indices[i]
            for z in range(len(drivers)):
                if z!=position:
                    adjacency_matrix_drivers[z][position]=adjacency_matrix_drivers[z][position]+1
    for j in teams_indices:
        if j==team1:
            p=teams_indices[j]
            for y in range(len(teams)):
                if y!=p:
                    adjacency_matrix_teams[y][p]=adjacency_matrix_teams[y][p]+1
            
            
            
#compute the out degree of the drivers                    
drivers_out_degrees = np.sum(adjacency_matrix_drivers, axis=1)

#compute the out degree of the teams
teams_out_degrees = np.sum(adjacency_matrix_teams, axis=1)

#adjacency_matrix is then divided by the number of out-degrees per driver to produce the hyperlink matrix
hyperlink_matrix_drivers = adjacency_matrix_drivers / drivers_out_degrees[:, np.newaxis]

#adjacency_matrix is then divided by the number of out-degrees per team to produce the hyperlink matrix
hyperlink_matrix_teams = adjacency_matrix_teams / teams_out_degrees[:, np.newaxis]

#sums of each row
row_sums = hyperlink_matrix_drivers.sum(axis=1)

row_sums_teams = hyperlink_matrix_teams.sum(axis=1)
#a is the dumping factor
a=0.85

#E is a [n×n] matrix populated entirely with the value 1/n
#compute the matrixes E both for teams and for drivers
E_drivers=np.zeros((len(drivers), len(drivers)))
for i in range(len(drivers)):
    for j in range(len(drivers)):
        E_drivers[i][j]=1/len(drivers)

E_teams=np.zeros((len(teams), len(teams)))
for i in range(len(teams)):
    for j in range(len(teams)):
        E_teams[i][j]=1/len(teams)
        
# Create the Google matrix
google_matrix_drivers = a * hyperlink_matrix_drivers + (1 - a) *E_drivers


google_matrix_teams = a * hyperlink_matrix_teams + (1 - a) *E_teams

# Initialize the PageRank scores for the drivers
initial_scores_drivers = np.ones(len(drivers)) / len(drivers)
current_scores_drivers = initial_scores_drivers

# Initialize the PageRank scores for the teams
initial_scores_teams= np.ones(len(teams)) / len(teams)
current_scores_teams = initial_scores_teams

conv_threshold = 0.0001
#num_iterations = 10  # Adjust the number of iterations as needed
#for _ in range(num_iterations):
    #current_scores = np.dot(current_scores, google_matrix)
    
    
num_iterations = 0
previous_scores = np.zeros(len(drivers))
while np.max(np.abs(current_scores_drivers - previous_scores)) > conv_threshold:
    previous_scores = current_scores_drivers
    current_scores_drivers = np.dot(current_scores_drivers, google_matrix_drivers)
    num_iterations = num_iterations + 1

# Sort the drivers and teams based on the final PageRank scores
sorted_drivers = [driver for _, driver in sorted(zip(current_scores_drivers, drivers), reverse=True)]


num_iterations1 = 0
previous_scores_teams = np.zeros(len(teams))
while np.max(np.abs(current_scores_teams - previous_scores_teams)) > conv_threshold:
    previous_scores_teams = current_scores_teams
    current_scores_teams = np.dot(current_scores_teams, google_matrix_teams)
    num_iterations1 = num_iterations1 + 1
    
# Sort the teams and teams based on the final PageRank scores
sorted_teams = [team for _, team in sorted(zip(current_scores_teams, teams), reverse=True)]

print("The driver rankings are:")
for rank, driver in enumerate(sorted_drivers):
    print(f"{rank+1}. {driver}")

print("\n The Team Rankings are:")
for rank, team in enumerate(sorted_teams):
    print(f"{rank+1}. {team}")

