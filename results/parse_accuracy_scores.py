import re
import pandas as pd

f=open("js.txt", "r")
text = f.read()

x = re.findall(" [0-9]*/[0-9]* ", text)
print(x)



count = 1
scores = []
for i in x:
    if count >1 and count % 14 == 0:
        scores.append(int(i[0:-6])/100)
    count += 1

result = pd.DataFrame(scores)
result.T.to_csv('result_temp.csv', encoding='utf-8', index=False, header=None)
