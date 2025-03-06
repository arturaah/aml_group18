import re
import pandas as pd
import numpy as np
def extract_elbo_values(csv_file):
    elbo_dict = {}
    with open(csv_file, 'r') as f:
        for line in f:
            match = re.match(r"Epoch (\d+): ELBO: (-?\d+\.\d+)", line)
            if match:
                epoch, elbo = int(match.group(1)), float(match.group(2))
                elbo_dict[epoch] = elbo
    return elbo_dict


def main():
    epochs = list(range(5, 41, 5))  # Epochs 5 to 40 in steps of 5
    priors = ["gaussian", "mog", "vamp"]
    seeds = range(1, 6)  # Seeds 1 to 5

    # Initialize dictionary with empty lists for ELBO values
    elbo_data = {prior: {f"epoch_{epoch}": [] for epoch in epochs} for prior in priors}

    # Populate ELBO values
    for prior in priors:
        for seed in seeds:
            csv_file = f"VAE/trained_models/{prior}_{seed}/ELBO.csv"
            elbo_dict_single = extract_elbo_values(csv_file)

            for epoch in epochs:
                if epoch in elbo_dict_single:
                    elbo_data[prior][f"epoch_{epoch}"].append(elbo_dict_single[epoch])

    # Compute Mean and Std
    elbo_stats = {
        prior: {
            epoch: {
                "mean": np.mean(values) if values else None,
                "std": np.std(values, ddof=1) if len(values) > 1 else None
            } 
            for epoch, values in elbo_data[prior].items()
        }
        for prior in priors
    }
    # display in a pandas dataframe
    elbo_stats = pd.DataFrame(elbo_stats).round(2)
    # Save to CSV
    elbo_stats.to_csv("VAE/trained_models/elbo_stats.csv")
    #return elbo_stats

# Run the main function
elbo_stats = main()

