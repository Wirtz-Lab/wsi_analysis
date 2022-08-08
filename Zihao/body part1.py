import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
df = pd.read_excel(r'\\fatherserverdw\kyuex\imlist_all.xlsx')

print(df['body part 1'])
body = df['body part 1']

print(Counter(body).keys())
print(Counter(body).values())
bodyparts = list(Counter(body).keys())
values = list(Counter(body).values())


plt.bar(bodyparts, values)
plt.show()
