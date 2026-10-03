import random

import numpy as np
import torch
import anny

from measurements import BodyMeasurements


DT = torch.float64

# Age is fixed according to Stage B requirements.
AGE = 0.5

RANDOM_TRIALS = 30
LOCAL_ITERATIONS = 40

LOW = 0.001
HIGH = 0.999

# Height and weight must be matched strongly.
HEIGHT_WEIGHT_IMPORTANCE = 20.0


def make_phenotype(
    model,
    gender,
    height,
    weight,
    muscle,
    proportions,
):
    ph = {
        key: torch.full(
            (1,),
            0.5,
            dtype=DT,
        )
        for key in model.phenotype_labels
    }

    ph["gender"] = torch.full(
        (1,),
        gender,
        dtype=DT,
    )

    ph["age"] = torch.full(
        (1,),
        AGE,
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
    height,
    weight,
    muscle,
    proportions,
):
    phenotype = make_phenotype(
        model,
        gender,
        height,
        weight,
        muscle,
        proportions,
    )

    output = model(
        phenotype_kwargs=phenotype
    )

    result = measurements.measure(
        output
    )

    return output, result


def get_values(result):
    height = result.get("height_m")
    mass = result.get("mass_kg")
    waist = result.get("waist_m")
    chest = result.get("chest_m")
    hip = result.get("hip_m")

    if None in (
        height,
        mass,
        waist,
        chest,
        hip,
    ):
        return None

    return np.array(
        [
            float(height),
            float(mass),
            float(waist),
            float(chest),
            float(hip),
        ],
        dtype=np.float64,
    )


def get_targets(
    height_cm,
    weight_kg,
    waist_cm,
    chest_cm,
    hip_cm,
):
    return np.array(
        [
            height_cm / 100.0,
            weight_kg,
            waist_cm / 100.0,
            chest_cm / 100.0,
            hip_cm / 100.0,
        ],
        dtype=np.float64,
    )


def residuals(
    result,
    targets,
):
    values = get_values(result)

    if values is None:
        return None

    return (
        values - targets
    ) / targets


def score(
    result,
    targets,
):
    r = residuals(
        result,
        targets,
    )

    if r is None:
        return float("inf")

    # Strong priority for exact height and weight.
    height_weight_error = (
        r[0] ** 2
        + r[1] ** 2
    )

    # Body-shape measurements.
    body_error = (
        r[2] ** 2
        + r[3] ** 2
        + r[4] ** 2
    )

    return float(
        HEIGHT_WEIGHT_IMPORTANCE
        * height_weight_error
        + body_error
    )


def evaluate_params(
    model,
    measurements,
    gender,
    params,
):
    height = params[0]
    weight = params[1]
    muscle = params[2]
    proportions = params[3]

    _, result = evaluate(
        model,
        measurements,
        gender,
        height,
        weight,
        muscle,
        proportions,
    )

    return result


def coarse_search(
    model,
    measurements,
    gender,
    targets,
):
    print()
    print("COARSE RANDOM SEARCH")
    print("--------------------")

    best_params = None
    best_result = None
    best_score = float("inf")

    for i in range(
        RANDOM_TRIALS
    ):
        params = np.array(
            [
                random.uniform(
                    LOW,
                    HIGH,
                ),
                random.uniform(
                    LOW,
                    HIGH,
                ),
                random.uniform(
                    LOW,
                    HIGH,
                ),
                random.uniform(
                    LOW,
                    HIGH,
                ),
            ],
            dtype=np.float64,
        )

        result = evaluate_params(
            model,
            measurements,
            gender,
            params,
        )

        current_score = score(
            result,
            targets,
        )

        if not np.isfinite(
            current_score
        ):
            print(
                f"  trial {i + 1:02d}"
                f" | invalid measurement -> skipped"
            )
            continue

        if current_score < best_score:
            best_score = current_score
            best_params = params.copy()
            best_result = result

            print(
                f"  trial {i + 1:02d}"
                f" | score={best_score:.6f}"
                f" | height={params[0]:.3f}"
                f" | weight={params[1]:.3f}"
                f" | muscle={params[2]:.3f}"
                f" | proportions={params[3]:.3f}"
            )

    if best_params is None:
        raise RuntimeError(
            "Stage B could not find any valid "
            "body shape with measurable "
            "height, weight, waist, chest and hip."
        )

    return (
        best_params,
        best_result,
        best_score,
    )


def local_refinement(
    model,
    measurements,
    gender,
    targets,
    initial_params,
):
    print()
    print("LOCAL REFINEMENT")
    print("----------------")

    params = initial_params.copy()

    step = 0.10

    best_result = evaluate_params(
        model,
        measurements,
        gender,
        params,
    )

    best_score = score(
        best_result,
        targets,
    )

    for iteration in range(
        LOCAL_ITERATIONS
    ):
        improved = False

        for index in range(4):

            for direction in [
                -1.0,
                1.0,
            ]:
                candidate = params.copy()

                candidate[index] += (
                    direction * step
                )

                candidate[index] = np.clip(
                    candidate[index],
                    LOW,
                    HIGH,
                )

                result = evaluate_params(
                    model,
                    measurements,
                    gender,
                    candidate,
                )

                current_score = score(
                    result,
                    targets,
                )

                if not np.isfinite(
                    current_score
                ):
                    continue

                if current_score < best_score:
                    params = candidate
                    best_result = result
                    best_score = current_score
                    improved = True

        if not improved:
            step *= 0.5

        if iteration % 5 == 0:
            print(
                f"  iteration {iteration + 1:02d}"
                f" | score={best_score:.6f}"
                f" | step={step:.5f}"
            )

        if step < 0.0005:
            break

    return (
        params,
        best_result,
        best_score,
    )


def print_results(
    params,
    result,
    targets,
):
    values = get_values(result)

    if values is None:
        raise RuntimeError(
            "Final Stage B result has "
            "invalid measurements."
        )

    target_height = targets[0]
    target_weight = targets[1]
    target_waist = targets[2]
    target_chest = targets[3]
    target_hip = targets[4]

    print()
    print("==============================")
    print("STAGE B - FIT RESULT")
    print("==============================")

    print()
    print("PHENOTYPE")
    print("---------")

    print(
        f"height      : {params[0]:.6f}"
    )

    print(
        f"weight      : {params[1]:.6f}"
    )

    print(
        f"muscle      : {params[2]:.6f}"
    )

    print(
        f"proportions : {params[3]:.6f}"
    )

    print(
        f"age         : {AGE:.6f}"
    )

    print()
    print("TARGET vs ACHIEVED")
    print("------------------")

    print(
        f"Height : "
        f"{target_height * 100:.2f} cm"
        f" -> "
        f"{values[0] * 100:.2f} cm"
        f" | error "
        f"{(values[0] - target_height) * 100:+.2f} cm"
    )

    print(
        f"Weight : "
        f"{target_weight:.2f} kg"
        f" -> "
        f"{values[1]:.2f} kg"
        f" | error "
        f"{values[1] - target_weight:+.2f} kg"
    )

    print(
        f"Waist  : "
        f"{target_waist * 100:.2f} cm"
        f" -> "
        f"{values[2] * 100:.2f} cm"
        f" | error "
        f"{(values[2] - target_waist) * 100:+.2f} cm"
    )

    print(
        f"Chest  : "
        f"{target_chest * 100:.2f} cm"
        f" -> "
        f"{values[3] * 100:.2f} cm"
        f" | error "
        f"{(values[3] - target_chest) * 100:+.2f} cm"
    )

    print(
        f"Hip    : "
        f"{target_hip * 100:.2f} cm"
        f" -> "
        f"{values[4] * 100:.2f} cm"
        f" | error "
        f"{(values[4] - target_hip) * 100:+.2f} cm"
    )

    print()
    print("RESIDUAL ERROR")
    print("--------------")

    print(
        f"Height : "
        f"{abs(values[0] - target_height) * 100:.2f} cm"
    )

    print(
        f"Weight : "
        f"{abs(values[1] - target_weight):.2f} kg"
    )

    print(
        f"Waist  : "
        f"{abs(values[2] - target_waist) * 100:.2f} cm"
    )

    print(
        f"Chest  : "
        f"{abs(values[3] - target_chest) * 100:.2f} cm"
    )

    print(
        f"Hip    : "
        f"{abs(values[4] - target_hip) * 100:.2f} cm"
    )

    print()
    print("FINAL SCORE")
    print("-----------")

    print(
        f"{score(result, targets):.8f}"
    )

    print()
    print("STAGE B COMPLETE")


def main():
    print("Starting Stage B...")

    model = anny.Anny()

    measurements = BodyMeasurements(
        model
    )

    # Example user.
    # We will replace these with real frontend values later.
    gender_name = "female"

    height_cm = 162.6
    weight_kg = 55.0

    waist_cm = 71.8
    chest_cm = 87.1
    hip_cm = 92.2

    gender = (
        1.0
        if gender_name == "female"
        else 0.0
    )

    targets = get_targets(
        height_cm,
        weight_kg,
        waist_cm,
        chest_cm,
        hip_cm,
    )

    print()
    print("TARGET USER")
    print("-----------")

    print(
        f"Gender : {gender_name}"
    )

    print(
        f"Height : {height_cm:.2f} cm"
    )

    print(
        f"Weight : {weight_kg:.2f} kg"
    )

    print(
        f"Waist  : {waist_cm:.2f} cm"
    )

    print(
        f"Chest  : {chest_cm:.2f} cm"
    )

    print(
        f"Hip    : {hip_cm:.2f} cm"
    )

    best_params, best_result, best_score = (
        coarse_search(
            model,
            measurements,
            gender,
            targets,
        )
    )

    print()
    print(
        f"Best coarse score: "
        f"{best_score:.8f}"
    )

    final_params, final_result, final_score = (
        local_refinement(
            model,
            measurements,
            gender,
            targets,
            best_params,
        )
    )

    print_results(
        final_params,
        final_result,
        targets,
    )


if __name__ == "__main__":
    main()