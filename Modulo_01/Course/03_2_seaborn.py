import matplotlib.pyplot as plt
import seaborn as sns

df = sns.load_dataset("iris")
sns.scatterplot(data=df, x="sepal_length", y="sepal_width", hue="species")
plt.show()
