import numpy as np
import matplotlib.pyplot as plt

# Générer des données simulées (log-normales)
np.random.seed(47)
x = np.sort(np.random.lognormal(mean=4, sigma=0.6, size=30))
cdf = np.arange(1, len(x) + 1) / len(x)  # CDF empirique

# Tracer la figure
plt.figure()

# Tracer la fonction de répartition cumulative (CDF)
plt.step(
    x, cdf, where="post", label="Fonction de répartition cumulative (CDF)", linewidth=2
)

# Tracer les échantillons observés comme des traits verticaux
for xi in x:
    plt.plot([xi, xi], [-0.02, 0], color="black", linewidth=1.5)

# Habillage en français
plt.title("Lien entre les valeurs détectées et la CDF")
plt.xlabel("Concentration (ppm)")
plt.ylabel("$P(X ≤ x)$")
plt.ylim(-0.05, 1.05)
plt.grid(ls=":")
plt.legend()
plt.tight_layout()
plt.savefig(f"CDF-example.png", dpi=300, bbox_inches="tight")
plt.show()
