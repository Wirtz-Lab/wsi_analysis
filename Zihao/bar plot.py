import pandas as pd
import numpy as np
import matplotlib.plts plt
from collections import Counter
df = pd.read_excel(r'\\fatherserverdw\kyuex\imlist_all.xlsx')

print(df['body part 1'])
body = df['body part 1']

print(Counter(body).keys())
print(Counter(body).values())

plt.bar(Counter(body).keys(), Counter(body).values, color = 'blue', width = 0.4)