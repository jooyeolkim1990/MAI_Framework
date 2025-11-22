
import pandas as pd
import matplotlib.pyplot as plt

# Load semantic convergence data
df = pd.read_csv("data/MAI_Cross-Round_Semantic_Convergence.csv")

# Plot the semantic similarity over mapping steps
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(df)+1),
         df["Semantic_Similarity"],
         marker="o",
         linewidth=2)
plt.title("MAI Cross-Round Semantic Convergence", fontsize=14)
plt.xlabel("Mapping Step", fontsize=12)
plt.ylabel("Semantic Similarity", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
