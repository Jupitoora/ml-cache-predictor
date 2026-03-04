import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from src.feature_engineering import load_dataset, preprocess_data
from src.cache_simulator import simulate_lru, simulate_ml


cache_sizes = [64, 128, 256, 512]

lru_results = []
ml_results = []

df = load_dataset()

X, y = preprocess_data(df)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

model.fit(X, y)

for size in cache_sizes:

    lru_hits, lru_misses = simulate_lru(df, cache_size=size)
    ml_hits, ml_misses = simulate_ml(df, model, cache_size=size)

    lru_hit_rate = lru_hits / (lru_hits + lru_misses)
    ml_hit_rate = ml_hits / (ml_hits + ml_misses)

    lru_results.append(lru_hit_rate)
    ml_results.append(ml_hit_rate)

    print(f"Cache Size {size}")
    print("LRU Hit Rate:", lru_hit_rate)
    print("ML Hit Rate:", ml_hit_rate)
    print()


plt.figure(figsize=(8,5))

plt.plot(cache_sizes, lru_results, marker="o", linewidth=2, label="LRU")
plt.plot(cache_sizes, ml_results, marker="o", linewidth=2, label="ML Predictor")

plt.xlabel("Cache Size (Blocks)")
plt.ylabel("Cache Hit Rate")

plt.title("Cache Hit Rate vs Cache Size")

plt.grid(True)

plt.legend()

plt.savefig("results/plots/cache_size_experiment.png", dpi=300)

plt.show()