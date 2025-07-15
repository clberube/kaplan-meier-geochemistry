import numpy as np
import matplotlib.pyplot as plt

# Générer des données simulées
np.random.seed(47)
n_total = 1000
valeurs_reelles = np.random.lognormal(mean=4, sigma=0.3, size=n_total)
lod = 50  # Limite de détection

# Identifier les valeurs sous la LOD
valeurs_detectees = valeurs_reelles[valeurs_reelles >= lod]
valeurs_censees = np.full(150, lod)  # 150 valeurs simulées comme < LOD

# Combiner pour l'affichage
donnees = np.concatenate([valeurs_censees, valeurs_detectees])

# Tracer l'histogramme
plt.figure()
plt.hist(donnees, bins=40, edgecolor="black", alpha=0.5)
plt.axvline(lod, color="C3", linestyle="--", lw=2, label="Limite de détection (LOD)")
plt.title("Données géochimiques avec valeurs < LOD")
plt.xlabel("Concentration (ppm)")
plt.ylabel("Fréquence")
plt.legend()
# plt.grid(ls=":")
plt.tight_layout()
plt.savefig(f"LOD-example.png", dpi=300, bbox_inches="tight")
plt.show()
