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

    gender = 1.0

    print()
    print("CHEST CONTINUITY TEST")
    print("----------------------")
    print()

    print(
        f"{'Weight':>8}"
        f"{'Chest':>12}"
        f"{'Chest Z':>12}"
        f"{'Change':>12}"
    )

    print("-" * 46)

    previous_chest = None

    for i in range(21):

        weight = i * 0.05

        ph = make_phenotype(
            model,
            gender,
            weight,
        )

        output = model(
            phenotype_kwargs=ph
        )

        result = measurements.measure(output)

        chest = result["chest_m"] * 100

        chest_info = measurements._chest_section(
            output["rest_vertices"]
        )

        chest_z = chest_info["z"]

        if previous_chest is None:
            change = 0.0
        else:
            change = chest - previous_chest

        print(
            f"{weight:>8.2f}"
            f"{chest:>12.2f}"
            f"{chest_z:>12.4f}"
            f"{change:>12.2f}"
        )

        previous_chest = chest


if __name__ == "__main__":
    main()
