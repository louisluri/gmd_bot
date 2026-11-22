import pandas as pd
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6), facecolor="grey")


model = "pg_ffnn00"
levels = ["stereo_madness"]
colors = ['indigo', 'g', "#8A2BE2", 'b', 'c', 'g', 'y', 'orange', 'r', 'darkred']

window = 50
for i in range(len(levels)):
    df = pd.read_csv(f"saves/{model}/{levels[i]}/metrics.csv")

    df['rolling_avg'] = df['percent'].rolling(window=window, center=True).mean()

    plt.plot(df['step'], df['percent'], alpha=0.2, color=colors[i])
    plt.plot(df['step'], df['rolling_avg'], color=colors[i], linewidth=2, label=levels[i])

plt.xlabel('step')
plt.ylabel('percent')
plt.title(model)
#plt.ylim(-350, 350)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()