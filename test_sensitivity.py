import torch
import anny

from measurements import BodyMeasurements


DT = torch.float64


def make_phenotype(model, gender, weight):
    ph = {
        k: torch.full((1,), 0.5, dtype=DT)
        for k in model.phenotype_labels
    }

    ph["gender"] = torch.full((1,), gender, dtype=DT)
    ph["age"] = torch.full((1,), 0.5, dtype=DT)
    ph["weight"] = torch.full((1,), weight, dtype=DT)

    return ph


def main():
    print("Loading Anny...")

    model = anny.Anny()
    measurements = BodyMeasurements(model)

    genders = [
        ("Male", 0.0),
        ("Middle", 0.5),
        ("Female", 1.0),
    ]

    weights = [
        ("Light", 0.0),
        ("Average", 0.5),
        ("Heavy", 1.0),
    ]

    print()
    print("SENSITIVITY TEST")
    print("----------------")
    print()

    print(
        f"{'Gender':<10}"
        f"{'Weight':<10}"
        f"{'Waist':>10}"
        f"{'Chest':>10}"
        f"{'Hip':>10}"
    )

    print("-" * 50)

    for gender_name, gender_value in genders:
        for weight_name, weight_value in weights:

            ph = make_phenotype(
                model,
                gender_value,
                weight_value,
            )

            output = model(
                phenotype_kwargs=ph
            )

            result = measurements.measure(output)

            waist = result["waist_m"] * 100
            chest = result["chest_m"] * 100
            hip = result["hip_m"] * 100

            print(
                f"{gender_name:<10}"
                f"{weight_name:<10}"
                f"{waist:>10.2f}"
                f"{chest:>10.2f}"
                f"{hip:>10.2f}"
            )


if __name__ == "__main__":
    main()