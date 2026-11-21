import seaborn as sns

df = sns.load_dataset("penguins")

df = df.dropna()

x = df[["bill_lenght_mm", "bill_depth_mm"]]
y = df["species"]

models = {
    "KNN": make_pipeline()
}
