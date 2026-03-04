from collections import OrderedDict
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.feature_engineering import load_dataset, preprocess_data


class LRUCache:

    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.hits = 0
        self.misses = 0

    def access(self, address):

        if address in self.cache:
            self.hits += 1
            self.cache.move_to_end(address)

        else:
            self.misses += 1

            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)

            self.cache[address] = True


def simulate_lru(trace, cache_size=256):

    cache = LRUCache(cache_size)

    for address in trace["address"]:
        cache.access(address)

    return cache.hits, cache.misses


def simulate_ml(trace, model, cache_size=256):

    cache = OrderedDict()
    hits = 0
    misses = 0

    for _, row in trace.iterrows():

        address = row["address"]

        features = np.array([
            row["pc"],
            row["address"],
            row["reuse_distance"],
            row["frequency"],
            row["gap"],
            row["access_type"]
        ])

        prediction = model.predict([features])[0]

        if address in cache:
            hits += 1
            cache.move_to_end(address)

        else:
            misses += 1

            if len(cache) >= cache_size:

                if prediction == 1:
                    cache.popitem(last=False)

            cache[address] = True

    return hits, misses


def simulate_hybrid(trace, model, cache_size=256):

    cache = OrderedDict()
    hits = 0
    misses = 0

    for _, row in trace.iterrows():

        address = row["address"]

        features = np.array([
            row["pc"],
            row["address"],
            row["reuse_distance"],
            row["frequency"],
            row["gap"],
            row["access_type"]
        ])

        ml_prediction = model.predict([features])[0]

        heuristic_prediction = 1 if row["reuse_distance"] <= 10 else 0

        useful = 1 if (ml_prediction or heuristic_prediction) else 0

        if address in cache:
            hits += 1
            cache.move_to_end(address)

        else:
            misses += 1

            if len(cache) >= cache_size:

                if useful == 1:
                    cache.popitem(last=False)

            cache[address] = True

    return hits, misses


if __name__ == "__main__":

    # Load dataset
    df = load_dataset()

    # Prepare features
    X, y = preprocess_data(df)

    # Train Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    model.fit(X, y)

    # Run simulations
    lru_hits, lru_misses = simulate_lru(df)
    ml_hits, ml_misses = simulate_ml(df, model)
    hybrid_hits, hybrid_misses = simulate_hybrid(df, model)

    print("\n===== Cache Simulation Results =====")

    print("\nLRU Policy")
    print("Hits:", lru_hits)
    print("Misses:", lru_misses)

    print("\nML Policy (Random Forest)")
    print("Hits:", ml_hits)
    print("Misses:", ml_misses)

    print("\nHybrid ML + Heuristic Policy")
    print("Hits:", hybrid_hits)
    print("Misses:", hybrid_misses)

    lru_hit_rate = lru_hits / (lru_hits + lru_misses)
    ml_hit_rate = ml_hits / (ml_hits + ml_misses)
    hybrid_hit_rate = hybrid_hits / (hybrid_hits + hybrid_misses)

    print("\nLRU Hit Rate:", lru_hit_rate)
    print("ML Hit Rate:", ml_hit_rate)
    print("Hybrid Hit Rate:", hybrid_hit_rate)