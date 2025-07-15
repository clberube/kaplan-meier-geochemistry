import numpy as np
import matplotlib.pyplot as plt

# Paramètres
np.random.seed(42)
lod = 10  # Limite de détection

# Générer une distribution log-normale simulée
x = np.sort(np.random.lognormal(mean=4, sigma=0.3, size=1000))
cdf = np.arange(1, len(x) + 1) / len(x)

# Si aucune valeur sous le LOD, on en ajoute manuellement
if np.sum(x <= lod) < 10:
    x = np.concatenate([np.random.uniform(1, lod, size=50), x])
    x.sort()
    cdf = np.arange(1, len(x) + 1) / len(x)

# Identifier la zone < LOD
mask_lod = x <= lod
x_lod = x[mask_lod]
cdf_lod = cdf[mask_lod]

# Simuler des valeurs imputées dans la zone < LOD
imputed_values = np.random.uniform(x_lod.min(), x_lod.max(), size=8)
imputed_values.sort()

# Tracer la figure
plt.figure()
plt.step(x, cdf, where="post", label="CDF estimée", color="C0")
plt.axvline(lod, color="gray", linestyle="--", label="LOD")
plt.fill_between(x_lod, 0, cdf_lod, color="lightblue", alpha=0.5, label="Zone < LOD")
plt.vlines(imputed_values, 0, -0.02, color="red", linewidth=2, label="Valeurs imputées")
# plt.scatter(imputed_values, [0.02] * len(imputed_values), color="red")

# Ligne verticale à la LOD

# Habillage en français
plt.xlabel("Concentration (ppm)")
plt.ylabel(r"$P(X \leq x)$")
plt.title("Imputation des valeurs < LOD à partir de la CDF")
plt.legend()
plt.grid(ls=":")
plt.tight_layout()
plt.xscale("log")
plt.savefig(f"imputation-example.png", dpi=300, bbox_inches="tight")
plt.show()
