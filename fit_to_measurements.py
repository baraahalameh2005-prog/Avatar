import argparse
import json
import time

import numpy as np
import torch
import anny


DT = torch.float64


def make_phenotype(
    model,
    gender,
    age,
    height,
    weight,
    muscle,
    proportions,
):
    ph = {
        k: torch.full(
            (1,),
            0.5,
            dtype=DT,
        )
        for k in model.phenotype_labels
    }

    ph["gender"] = torch.full(
        (1,),
        gender,
        dtype=DT,
    )

    ph["age"] = torch.full(
        (1,),
        age,
        dtype=DT,
    )

    ph["height"] = torch.full(
        (1,),
        height,
        dtype=DT,
    )

    ph["weight"] = torch.full(
        (1,),
        weight,
        dtype=DT,
    )

    ph["muscle"] = torch.full(
        (1,),
        muscle,
        dtype=DT,
    )

    ph["proportions"] = torch.full(
        (1,),
        proportions,
        dtype=DT,
    )

    return ph


# ============================================================
# FULL EVALUATION
# ============================================================

def evaluate(
    model,
    measurements,
    gender,
    age,
    height,
    weight,
    muscle,
    proportions,
):
    ph = make_phenotype(
        model,
        gender,
        age,
        height,
        weight,
        muscle,
        proportions,
    )

    output = model(
        phenotype_kwargs=ph
    )

    result = measurements.measure(
        output
    )

    return output, result


# ============================================================
# LIGHT EVALUATION
# ============================================================

def evaluate_anthropometry(
    model,
    measurements,
    gender,
    age,
    height,
    weight,
    muscle,
    proportions,
):
    ph = make_phenotype(
        model,
        gender,
        age,
        height,
        weight,
        muscle,
        proportions,
    )

    output = model(
        phenotype_kwargs=ph
    )

    base = measurements.anthropometry(
        output["rest_vertices"]
    )

    height_m = float(
        base["height"].item()
    )

    mass_kg = float(
        base["mass"].item()
    )

    return output, height_m, mass_kg


# ============================================================
# SOLVE HEIGHT
# ============================================================

def solve_height(
    model,
    measurements,
    gender,
    age,
    weight,
    muscle,
    proportions,
    target_height,
):
    lo = 0.001
    hi = 0.999

    # Reduced from 15 to 10 iterations for speed testing.
    for _ in range(10):
        mid = (lo + hi) / 2.0

        _, current_height, _ = evaluate_anthropometry(
            model,
            measurements,
            gender,
            age,
            mid,
            weight,
            muscle,
            proportions,
        )

        if current_height < target_height:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2.0


# ============================================================
# SOLVE WEIGHT
# ============================================================

def solve_weight(
    model,
    measurements,
    gender,
    age,
    height,
    muscle,
    proportions,
    target_weight,
):
    lo = 0.001
    hi = 0.999

    # Reduced from 15 to 10 iterations for speed testing.
    for _ in range(10):
        mid = (lo + hi) / 2.0

        _, _, current_mass = evaluate_anthropometry(
            model,
            measurements,
            gender,
            age,
            height,
            mid,
            muscle,
            proportions,
        )

        if current_mass < target_weight:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2.0


# ============================================================
# EVALUATE BODY SHAPE
# ============================================================

def evaluate_shape(
    model,
    measurements,
    gender,
    target_height,
    target_weight,
    age,
    muscle,
    proportions,
):
    initial_weight = 0.5

    # Step 1: solve height
    height = solve_height(
        model,
        measurements,
        gender,
        age,
        initial_weight,
        muscle,
        proportions,
        target_height,
    )

    # Step 2: solve weight
    weight = solve_weight(
        model,
        measurements,
        gender,
        age,
        height,
        muscle,
        proportions,
        target_weight,
    )

    # Step 3: solve height again after weight is known
    height = solve_height(
        model,
        measurements,
        gender,
        age,
        weight,
        muscle,
        proportions,
        target_height,
    )

    # Step 4: ONLY NOW calculate all body measurements
    output, result = evaluate(
        model,
        measurements,
        gender,
        age,
        height,
        weight,
        muscle,
        proportions,
    )

    return output, result, height, weight


# ============================================================
# FIT BODY SHAPE
# ============================================================

def fit(
    model,
    measurements,
    gender,
    height_cm,
    weight_kg,
    waist_cm,
    chest_cm,
    hip_cm,
):
    try:
        from scipy.optimize import least_squares
    except ImportError:
        raise RuntimeError(
            "scipy is required. Install it with: "
            "python -m pip install scipy"
        )

    target_height = height_cm / 100.0
    target_weight = weight_kg

    targets = np.array(
        [
            waist_cm / 100.0,
            chest_cm / 100.0,
            hip_cm / 100.0,
        ],
        dtype=np.float64,
    )

    # age, muscle, proportions
    x0 = np.array(
        [
            0.5,
            0.5,
            0.5,
        ],
        dtype=np.float64,
    )

    lower = np.array(
        [
            0.001,
            0.001,
            0.001,
        ],
        dtype=np.float64,
    )

    upper = np.array(
        [
            0.999,
            0.999,
            0.999,
        ],
        dtype=np.float64,
    )

    calls = [0]

    def objective(params):
        calls[0] += 1

        age = params[0]
        muscle = params[1]
        proportions = params[2]

        _, result, height, weight = evaluate_shape(
            model,
            measurements,
            gender,
            target_height,
            target_weight,
            age,
            muscle,
            proportions,
        )

        values = np.array(
            [
                float(result["waist_m"]),
                float(result["chest_m"]),
                float(result["hip_m"]),
            ],
            dtype=np.float64,
        )

        r = (
            values - targets
        ) / targets

        if calls[0] % 5 == 0:
            print(
                f"  iteration {calls[0]:>3}"
                f" | error {np.linalg.norm(r):.5f}"
                f" | age {age:.3f}"
                f" | muscle {muscle:.3f}"
                f" | proportions {proportions:.3f}"
            )

        return r

    print("Optimizing body shape...")

    solution = least_squares(
        objective,
        x0,
        bounds=(lower, upper),
        method="trf",
        xtol=1e-6,
        ftol=1e-6,
        gtol=1e-6,
        max_nfev=40,
        verbose=0,
    )

    age = solution.x[0]
    muscle = solution.x[1]
    proportions = solution.x[2]

    output, result, height, weight = evaluate_shape(
        model,
        measurements,
        gender,
        target_height,
        target_weight,
        age,
        muscle,
        proportions,
    )

    print()
    print("Solved parameters:")

    print(
        f"  Height      : {height:.4f}"
    )

    print(
        f"  Weight      : {weight:.4f}"
    )

    print(
        f"  Age         : {age:.4f}"
    )

    print(
        f"  Muscle      : {muscle:.4f}"
    )

    print(
        f"  Proportions : {proportions:.4f}"
    )

    return output, result, np.array(
        [
            height,
            weight,
            age,
            muscle,
            proportions,
        ],
        dtype=np.float64,
    )


# ============================================================
# FASTAPI REUSABLE FUNCTION
# ============================================================

def fit_body(
    model,
    measurements,
    gender,
    height_cm,
    weight_kg,
    waist_cm,
    chest_cm,
    hip_cm,
):
    """
    Reusable function for FastAPI.

    Takes the user's body measurements and returns
    the fitted Anny parameters and the generated body data.
    """

    # Start measuring the fitting time.
    start_time = time.perf_counter()

    gender_value = (
        1.0
        if gender == "female"
        else 0.0
    )

    output, result, params = fit(
        model=model,
        measurements=measurements,
        gender=gender_value,
        height_cm=height_cm,
        weight_kg=weight_kg,
        waist_cm=waist_cm,
        chest_cm=chest_cm,
        hip_cm=hip_cm,
    )

    # Calculate total fitting time.
    elapsed = time.perf_counter() - start_time

    print()

    print(
        f"Fitting time: {elapsed:.2f} seconds"
    )

    print(
        f"Fitting time: {elapsed / 60:.2f} minutes"
    )

    return {
        "output": output,
        "result": result,
        "parameters": {
            "gender": gender,
            "height": float(params[0]),
            "weight": float(params[1]),
            "age": float(params[2]),
            "muscle": float(params[3]),
            "proportions": float(params[4]),
        },
    }


# ============================================================
# COMMAND LINE
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gender",
        choices=["male", "female"],
        default="female",
    )

    parser.add_argument(
        "--height",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--weight",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--waist",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--chest",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--hip",
        type=float,
        required=True,
    )

    args = parser.parse_args()

    print("Loading Anny...")

    model = anny.Anny()

    from measurements import BodyMeasurements

    measurements = BodyMeasurements(model)

    print()
    print("FITTING BODY SHAPE")
    print("------------------")
    print()

    print("Target:")

    print(
        f"  Height : {args.height:.2f} cm"
    )

    print(
        f"  Weight : {args.weight:.2f} kg"
    )

    print(
        f"  Waist  : {args.waist:.2f} cm"
    )

    print(
        f"  Chest  : {args.chest:.2f} cm"
    )

    print(
        f"  Hip    : {args.hip:.2f} cm"
    )

    print()

    fitted = fit_body(
        model=model,
        measurements=measurements,
        gender=args.gender,
        height_cm=args.height,
        weight_kg=args.weight,
        waist_cm=args.waist,
        chest_cm=args.chest,
        hip_cm=args.hip,
    )

    result = fitted["result"]
    params = fitted["parameters"]

    print()
    print("RESULT")
    print("------")

    clean = {
        key: round(float(value), 4)
        for key, value in result.items()
        if key != "hip_z"
    }

    print(
        json.dumps(
            clean,
            indent=2,
        )
    )

    print()
    print("PARAMETERS")
    print("----------")

    print(
        json.dumps(
            {
                "height": round(
                    params["height"],
                    4,
                ),
                "weight": round(
                    params["weight"],
                    4,
                ),
                "age": round(
                    params["age"],
                    4,
                ),
                "muscle": round(
                    params["muscle"],
                    4,
                ),
                "proportions": round(
                    params["proportions"],
                    4,
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()