import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from tqdm import tqdm


def run_kaplan_meier_analysis(input_file, element, unit, bootstrap, impute):

    # Load data
    df = pd.read_csv(input_file)

    print("\n--- File name ---")
    print(Path(input_file))

    print("\n--- Element name ---")
    print(element)

    print("\n--- Original data description ---")
    print(df[element].describe())

    # Identify censored data
    df["is_censored"] = df[element] < 0
    df["value"] = df[element].abs()
    df["reversed"] = -df["value"]

    print("\n--- Number of censored values ---")
    print(df["is_censored"].sum(), "out of", df[element].count())

    # Define timeline from reversed values
    timeline = np.sort(-df["value"].values)

    # Fit Reverse Kaplan-Meier
    kmf = KaplanMeierFitter()
    kmf.fit(df["reversed"], event_observed=~df["is_censored"], timeline=timeline)

    # Reverse and sort survival function
    sf = kmf.survival_function_
    sf.index = -sf.index
    sf = sf.sort_index()
    sf["KM_estimate"] = sf["KM_estimate"].iloc[::-1].values
    sf["CDF"] = 1 - sf["KM_estimate"]

    # --- Statistics functions ---
    def rkm_mean(sf):
        return np.trapezoid(sf["KM_estimate"].values, sf.index.values)

    def rkm_percentile(sf, p):
        cdf_vals = 1 - sf["KM_estimate"]
        return np.interp(p, cdf_vals, sf.index)

    # --- Bootstrap procedure ---
    def bootstrap_rkm(df, n_boot=1000, timeline=None, random_state=42):
        rng = np.random.default_rng(random_state)
        means, medians, p25s, p75s, p01s, p99s = [], [], [], [], [], []

        detected = df[~df["is_censored"]]
        censored = df[df["is_censored"]]
        n_detected = len(detected)
        n_censored = len(censored)

        for _ in tqdm(range(n_boot), desc="Bootstrapping"):
            boot_censored = censored.sample(
                n=n_censored, replace=True, random_state=rng.integers(1e9)
            )
            boot_detected = detected.sample(
                n=n_detected, replace=True, random_state=rng.integers(1e9)
            )
            sample = pd.concat([boot_censored, boot_detected]).sample(
                frac=1, random_state=rng.integers(1e9)
            )
            sample["reversed"] = -sample["value"]

            kmf_boot = KaplanMeierFitter()
            kmf_boot.fit(
                sample["reversed"],
                event_observed=~sample["is_censored"],
                timeline=timeline,
            )
            sf_boot = kmf_boot.survival_function_
            sf_boot.index = -sf_boot.index
            sf_boot = sf_boot.sort_index()
            sf_boot["KM_estimate"] = sf_boot["KM_estimate"].iloc[::-1].values

            try:
                means.append(rkm_mean(sf_boot))
                medians.append(rkm_percentile(sf_boot, 0.5))
                p01s.append(rkm_percentile(sf_boot, 0.01))
                p25s.append(rkm_percentile(sf_boot, 0.25))
                p75s.append(rkm_percentile(sf_boot, 0.75))
                p99s.append(rkm_percentile(sf_boot, 0.99))
            except Exception:
                means.append(np.nan)
                medians.append(np.nan)
                p01s.append(np.nan)
                p25s.append(np.nan)
                p75s.append(np.nan)
                p99s.append(np.nan)

        return {
            "mean": np.array(means),
            "median": np.array(medians),
            "p01": np.array(p01s),
            "p25": np.array(p25s),
            "p75": np.array(p75s),
            "p99": np.array(p99s),
        }

    # --- Run bootstrapping ---
    print()
    boot_results = bootstrap_rkm(df, n_boot=bootstrap, timeline=timeline)

    # --- Summary with 95% confidence intervals ---
    def summarize(name, values):
        values = values[~np.isnan(values)]
        est = np.mean(values)
        ci_low = np.percentile(values, 2.5)
        ci_high = np.percentile(values, 97.5)
        formatted = f"{name:<12} = {est:8.3f}   (95% CI: {ci_low:8.3f} - {ci_high:8.3f})   {unit:>1}"
        print(formatted)

    print("\n--- Bootstrapped confidence intervals ---")
    summarize("Mean", boot_results["mean"])
    summarize("Median", boot_results["median"])
    summarize("1st pct", boot_results["p01"])
    summarize("25th pct", boot_results["p25"])
    summarize("75th pct", boot_results["p75"])
    summarize("99th pct", boot_results["p99"])

    # Sampling function: randomly sample from conditional distribution < DL
    concentrations = sf.index.values
    cdf = sf["CDF"].values

    def sample_below_DL(dl, n=1):
        mask = concentrations < dl
        if mask.sum() < 2:
            return np.full(n, dl / 2)  # fallback to midpoint
        x = concentrations[mask]
        p = cdf[mask]
        p = p - p.min()
        p = p / p.max()
        p_diff = np.diff(np.append(0, p))
        return np.random.choice(x, size=n, p=p_diff / p_diff.sum())

    if impute:
        path = Path(input_file)
        output_file = path.with_name(f"{path.stem}_imp{path.suffix}")
        # Apply imputation to censored values
        df_imputed = df.copy()
        df_imputed[element + "_imp"] = df_imputed[element].copy()
        for i, row in df[df["is_censored"]].iterrows():
            dl = row[f"value"]
            df_imputed.at[i, element + "_imp"] = sample_below_DL(dl, n=1)[0]

        # Optional: print summary
        print("\n--- Imputation complete ---")
        print(f"{df['is_censored'].sum()} values were imputed.")

        print("\n--- Imputed data description ---")
        print(df_imputed[element].describe())

        # Save or analyze the imputed dataset
        df_imputed[[element, element + "_imp", "is_censored"]].to_csv(
            output_file, index=False
        )

    # --- Plot survival and CDF with percentiles ---
    fig, ax = plt.subplots()
    ax.plot(sf.index, sf["CDF"], label="CDF", color="k")
    ax.axvline(
        np.nanmean(boot_results["p01"]), color="C0", linestyle=":", label="1er pct"
    )
    ax.axvline(
        np.nanmean(boot_results["p25"]), color="C1", linestyle="--", label="25e pct"
    )
    ax.axvline(
        np.nanmean(boot_results["median"]),
        color="gray",
        linestyle="--",
        label="50e pct",
    )
    ax.axvline(
        np.nanmean(boot_results["p75"]), color="C2", linestyle="--", label="75e pct"
    )
    ax.axvline(
        np.nanmean(boot_results["p99"]), color="C3", linestyle=":", label="99e pct"
    )
    ax.set_title("Kaplan-Meier CDF")
    ax.set_xlabel(f"Concentration en {element} ({unit})")
    ax.set_ylabel(r"$P(X \leq x)$")
    ax.legend()
    ax.grid(False)
    ax.set_xscale("log")

    plt.tight_layout()
    fig_name = f"{element}_KM-CDF.png"
    plt.savefig(fig_name, dpi=300, bbox_inches="tight")

    print("\n--- Figure saved as ---")
    print(fig_name)
    print("Close figure to exit program.")
    plt.show()

    print("\n--- END ---")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyse géochimique avec Kaplan-Meier"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Chemin vers le fichier CSV à traiter (ex: 'chemin/vers/csv_file.csv')",
    )
    parser.add_argument(
        "element", type=str, help="Nom de la colonne de données à traiter (ex: 'Ba')"
    )
    parser.add_argument(
        "--unit",
        type=str,
        default="ppm",
        help="Unité des analyses (défaut: 'ppm')",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Nombre de répétitions bootstrap (défaut: 1000)",
    )
    parser.add_argument(
        "--impute",
        action="store_true",
        help="Effectuer l'imputation des valeurs censurées (défaut: Non)",
    )
    args = parser.parse_args()

    # Exécuter le code original
    run_kaplan_meier_analysis(
        args.input_file,
        args.element,
        args.unit,
        args.bootstrap,
        args.impute,
    )


if __name__ == "__main__":
    main()
