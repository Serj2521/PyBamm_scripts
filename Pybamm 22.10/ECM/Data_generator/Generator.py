import os
os.system('cls' if os.name == 'nt' else 'clear')
import pandas as pd

import pandas as pd
import os

# Sample DataFrame
df = pd.DataFrame({
    'Name': ['John', 'Emily', 'Jack'],
    'Age': [28, 23, 35],
    'Gender': ['Male', 'Female', 'Male']
})

# Specify the directory and filename
directory = 'Pybamm_scripts\Pybamm 22.10\Driving_cycles'
filename = 'sample_ex_dataframe.csv'

# Get the current working directory and join it with the directory where you want to save the CSV file
cwd = os.getcwd()
path = os.path.join(cwd, directory)
print(path)
# Save the DataFrame to CSV in the specified directory
df.to_csv(f'{path}/{filename}', index=False)