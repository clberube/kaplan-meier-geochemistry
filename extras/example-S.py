import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Données fictives inversées triées
data = np.sort(-np.random.lognormal(mean=4, sigma=0.6, size=30))
censored = np.array(5 * [True, False, True, False, False, False])

# Tri croissant (car valeurs inversées)
order = np.argsort(data)
data = data[order]
censored = censored[order]

# Construction de la courbe de survie
n = len(data)
at_risk = n
survival = [1.0]
timeline = [data[0]]
current_survival = 1.0

for i in range(n):
    if not censored[i]:
        current_survival *= (at_risk - 1) / at_risk
    at_risk -= 1
    survival.append(current_survival)
    timeline.append(data[i])

# Préparer la figure
fig, ax = plt.subplots(dpi=300)
for xi in data:
    plt.plot([xi, xi], [-0.02, 0], color="black", linewidth=1.5)
ax.set_xlim(min(data) - 0.5, 0.5)
ax.set_ylim(-0.05, 1.05)
ax.set_xlabel("Valeur inversée (–ppm)")
ax.set_ylabel(r"Survie estimée $S(x)$")
ax.set_title("Construction progressive de la fonction de survie")
ax.grid(ls=":")

(line,) = ax.step([], [], where="post", color="C0", lw=2)
(dots,) = ax.plot([], [], ".", color="C0")


def init():
    line.set_data([], [])
    dots.set_data([], [])
    return line, dots


def update(i):
    line.set_data(timeline[: i + 1], survival[: i + 1])
    dots.set_data(timeline[: i + 1], survival[: i + 1])
    return line, dots


ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(timeline),
    init_func=init,
    blit=True,
    interval=800,
    repeat=False,
)

# Enregistrement au format MP4 (nécessite ffmpeg installé)
writer = animation.FFMpegWriter(
    fps=2, metadata=dict(artist="Kaplan-Meier"), bitrate=1800
)
ani.save("S-example.mp4", writer=writer, dpi=300)

print("Animation enregistrée sous 'fonction_survie_kaplan_meier.mp4'")
