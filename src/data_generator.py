import numpy as np
import random
import pandas as pd


def generate_memory_trace(n=10000):

    data = []

    address_last_seen = {}
    address_frequency = {}

    hot_region_start = 100
    hot_region_size = 50

    for t in range(n):

        # Program counter simulation
        pc = random.randint(0, 255)

        # Simulate program phases
        phase = (t // 2000) % 3

        if phase == 0:
            # hot memory region
            address = hot_region_start + random.randint(0, hot_region_size)

        elif phase == 1:
            # spatial locality (sequential access)
            address = (t % 100) + 200

        else:
            # random access region
            address = random.randint(0, 1023)

        access_type = random.choice([0, 1])

        # frequency tracking
        address_frequency[address] = address_frequency.get(address, 0) + 1
        frequency = address_frequency[address]

        # last access gap
        if address in address_last_seen:
            gap = t - address_last_seen[address]
        else:
            gap = 100

        address_last_seen[address] = t

        reuse_distance = min(gap, 50)

        label = 1 if reuse_distance <= 10 else 0

        data.append([
            pc,
            address,
            reuse_distance,
            frequency,
            gap,
            access_type,
            label
        ])

    columns = [
        "pc",
        "address",
        "reuse_distance",
        "frequency",
        "gap",
        "access_type",
        "label"
    ]

    df = pd.DataFrame(data, columns=columns)

    return df


if __name__ == "__main__":

    df1 = generate_memory_trace(10000)
    df1.to_csv("data/raw/workload_hot.csv", index=False)

    df2 = generate_memory_trace(10000)
    df2.to_csv("data/raw/workload_seq.csv", index=False)

    df3 = generate_memory_trace(10000)
    df3.to_csv("data/raw/workload_random.csv", index=False)

    print("Workloads generated successfully!")