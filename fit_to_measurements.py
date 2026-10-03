import argparse
import json

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

    for _ in range(30):
        mid = (lo + hi) / 2.0

        _, result = evaluate(
            model,
            measurements,
            gender,
            age,
            mid,
            weight,
            muscle,
            proportions,
        )

        current = float(
            result["height_m"]
        )

        if current < target_height:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2.0


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

    for _ in range(30):
        mid = (lo + hi) / 2.0

        _, result = evaluate(
            model,
            measurements,
            gender,
            age,
            height,
            mid,
            muscle,
            proportions,
        )

        current = float(
            result["mass_kg"]
        )

        if current < target_weight:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2.0


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

    # Re-solve height once more after the weight is known.
    # This keeps the final height accurate.
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

    print()
    print("FITTING BODY SHAPE")
    print("------------------")
    print()
    print("Target:")
    print(
        f"  Height : {height_cm:.2f} cm"
    )
    print(
        f"  Weight : {weight_kg:.2f} kg"
    )
    print(
        f"  Waist  : {waist_cm:.2f} cm"
    )
    print(
        f"  Chest  : {chest_cm:.2f} cm"
    )
    print(
        f"  Hip    : {hip_cm:.2f} cm"
    )
    print()

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
        max_nfev=80,
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

    gender = (
        1.0
        if args.gender == "female"
        else 0.0
    )

    output, result, params = fit(
        model,
        measurements,
        gender,
        args.height,
        args.weight,
        args.waist,
        args.chest,
        args.hip,
    )

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
                    float(params[0]), 4
                ),
                "weight": round(
                    float(params[1]), 4
                ),
                "age": round(
                    float(params[2]), 4
                ),
                "muscle": round(
                    float(params[3]), 4
                ),
                "proportions": round(
                    float(params[4]), 4
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
