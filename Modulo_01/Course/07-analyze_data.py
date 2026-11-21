import os.path
from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument('-f', '--file', type=str, required=True)
args = parser.parse_args()

filepath: str = args.file

if not os.path.exists(filepath):
    print("File not found")
    exit(1)

df = pd.read_csv(filepath)

df1 = df[df["CVE_ENT"] == 31]
df = df1[df1.index >= 13000]

x = df['PERIODO']
y = df['MEDIA']

serie = plt.figure(figsize=(35, 5))

plt.xticks(rotation=45)
plt.plot(x, y)
plt.ylabel("Media")
plt.xlabel("Periodo")
plt.title("Temperaturas medias en Yucatán 2017-2026")

max_idx = df['MEDIA'].resample('YE').idxmax()
print(max_idx)


