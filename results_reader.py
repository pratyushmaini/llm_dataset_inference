# go to p_values/{model}
# read every csv file
# print: {csv_name}: Number of values < 0.1 = {number of values < 0.1}

import sys
model_name = sys.argv[1]
import os
import pandas as pd

p_values_dir = f"p_values/{model_name}"
p_values = {}
for file in sorted(os.listdir(p_values_dir)):
    if file.endswith(".csv"):
        p_values[file] = pd.read_csv(f"{p_values_dir}/{file}")

for key in p_values:
    print(f"{key}: Number of values < 0.1 = {len(p_values[key][p_values[key]['p_value'] < 0.1])}")