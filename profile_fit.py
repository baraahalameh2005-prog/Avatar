import cProfile
import io
import pstats
import time

import anny

from fit_to_measurements import fit_body
from measurements import BodyMeasurements


# ============================================================
# TEST INPUT
# ============================================================

TARGET = {
    "gender": "female",
    "height_cm": 162.62,
    "weight_kg": 49.99,
    "waist_cm": 71.80,
    "chest_cm": 86.60,
    "hip_cm": 92.15,
}


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading Anny...")

model = anny.Anny()

print("Anny loaded.")

measurements = BodyMeasurements(model)


# ============================================================
# TIMING COUNTERS
# ============================================================

stats = {
    "calls": 0,
    "time": 0.0,
}


# Save original model call
_original_forward = anny.Anny.forward


def timed_forward(self, *args, **kwargs):
    start = time.perf_counter()

    result = _original_forward(
        self,
        *args,
        **kwargs,
    )

    stats["time"] += (
        time.perf_counter() - start
    )

    stats["calls"] += 1

    return result


# Replace Anny.forward temporarily
anny.Anny.forward = timed_forward


# ============================================================
# RUN FIT
# ============================================================

print()
print("Starting profiling...")
print("This may take several minutes.")
print()

profiler = cProfile.Profile()

start_total = time.perf_counter()

profiler.enable()

result = fit_body(
    model=model,
    measurements=measurements,
    gender=TARGET["gender"],
    height_cm=TARGET["height_cm"],
    weight_kg=TARGET["weight_kg"],
    waist_cm=TARGET["waist_cm"],
    chest_cm=TARGET["chest_cm"],
    hip_cm=TARGET["hip_cm"],
)

profiler.disable()

total_time = (
    time.perf_counter()
    - start_total
)


# ============================================================
# PRINT BASIC RESULTS
# ============================================================

print()
print("========================================")
print("PROFILING RESULT")
print("========================================")

print(
    f"Total wall time        : "
    f"{total_time:.2f} seconds"
)

print(
    f"Anny.forward calls     : "
    f"{stats['calls']}"
)

print(
    f"Anny.forward total     : "
    f"{stats['time']:.2f} seconds"
)

if stats["calls"] > 0:

    average = (
        stats["time"]
        / stats["calls"]
    )

    print(
        f"Average Anny call     : "
        f"{average * 1000:.2f} ms"
    )

    percentage = (
        stats["time"]
        / total_time
        * 100
    )

    print(
        f"Anny percentage       : "
        f"{percentage:.1f}%"
    )


# ============================================================
# cPROFILE RESULTS
# ============================================================

output = io.StringIO()

output.write(
    f"Total wall time        : "
    f"{total_time:.2f} seconds\n"
)

output.write(
    f"Anny.forward calls     : "
    f"{stats['calls']}\n"
)

output.write(
    f"Anny.forward total     : "
    f"{stats['time']:.2f} seconds\n"
)

if stats["calls"] > 0:

    average = (
        stats["time"]
        / stats["calls"]
    )

    percentage = (
        stats["time"]
        / total_time
        * 100
    )

    output.write(
        f"Average Anny call     : "
        f"{average * 1000:.2f} ms\n"
    )

    output.write(
        f"Anny percentage       : "
        f"{percentage:.1f}%\n"
    )


output.write(
    "\n"
    "========================================\n"
    "TOP 30 BY CUMULATIVE TIME\n"
    "========================================\n"
)


pstats.Stats(
    profiler,
    stream=output,
).sort_stats(
    "cumulative"
).print_stats(30)


output.write(
    "\n"
    "========================================\n"
    "TOP 30 BY SELF TIME\n"
    "========================================\n"
)


pstats.Stats(
    profiler,
    stream=output,
).sort_stats(
    "tottime"
).print_stats(30)


profile_text = output.getvalue()


print()
print(profile_text)


# ============================================================
# SAVE REPORT
# ============================================================

with open(
    "profile_baseline.txt",
    "w",
    encoding="utf-8",
) as file:

    file.write(profile_text)


print()
print(
    "Profile saved to:"
)

print(
    "profile_baseline.txt"
)