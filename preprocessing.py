import pandas as pd
data = pd.read_csv("Data/birds.csv")


print(data[data['class id'] == 0].to_markdown())