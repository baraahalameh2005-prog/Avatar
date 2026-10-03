import torch
import anny

from measurements import BodyMeasurements


DT = torch.float64


def make_phenotype(
    model,
    gender=1.0,
    age=0.5,
    height=0.5,
    weight=0.5,
    muscle=0.5,
    proportions=0.5,
):
    phenotype = {
        key: torch.full(
            (1,),
            0.5,
            dtype=DT,
        )
        for key in model.phenotype_labels
    }

    phenotype["gender"] = torch.full(
        (1,),
        gender,
        dtype=DT,
    )

    phenotype["age"] = torch.full(
        (1,),
        age,
        dtype=DT,
    )

    phenotype["height"] = torch.full(
        (1,),
        height,
        dtype=DT,
    )

    phenotype["weight"] = torch.full(
        (1,),
        weight,
        dtype=DT,
    )

    phenotype["muscle"] = torch.full(
        (1,),
        muscle,
        dtype=DT,
    )

    phenotype["proportions"] = torch.full(
        (1,),
        proportions,
        dtype=DT,
    )

    return phenotype


def measure_body(
    model,
    measurements,
    weight=0.5,
    muscle=0.5,
    proportions=0.5,
):
    phenotype = make_phenotype(
        model,
        weight=weight,
        muscle=muscle,
        proportions=proportions,
    )

    output = model(
        phenotype_kwargs=phenotype
    )

    result = measurements.measure(
        output
    )

    return {
        "height": result["height_m"] * 100,
        "weight": result["mass_kg"],
        "waist": result["waist_m"] * 100,
        "chest": (
            result["chest_m"] * 100
            if result["chest_m"] is not None
            else None
        ),
        "hip": (
            result["hip_m"] * 100
            if result["hip_m"] is not None
            else None
        ),
    }


def continuity_test(model, measurements):
    print()
    print("================================")
    print("STAGE A - CONTINUITY TEST")
    print("================================")
    print()

    previous = None
    max_jumps = {
        "waist": 0.0,
        "chest": 0.0,
        "hip": 0.0,
    }

    jump_records = []

    for i in range(21):
        weight = i * 0.05

        result = measure_body(
            model,
            measurements,
            weight=weight,
            muscle=0.5,
            proportions=0.5,
        )

        print(
            f"weight={weight:.2f} | "
            f"waist={result['waist']:.2f} cm | "
            f"chest={result['chest']:.2f} cm | "
            f"hip={result['hip']:.2f} cm"
        )

        if previous is not None:
            for name in [
                "waist",
                "chest",
                "hip",
            ]:
                current_value = result[name]
                previous_value = previous[name]

                if (
                    current_value is None
                    or previous_value is None
                ):
                    continue

                jump = abs(
                    current_value
                    - previous_value
                )

                if jump > max_jumps[name]:
                    max_jumps[name] = jump

                if jump > 3.0:
                    jump_records.append(
                        {
                            "weight_from": weight - 0.05,
                            "weight_to": weight,
                            "measurement": name,
                            "jump": jump,
                        }
                    )

        previous = result

    print()
    print("Maximum jumps:")
    print(
        f"Waist : {max_jumps['waist']:.2f} cm"
    )
    print(
        f"Chest : {max_jumps['chest']:.2f} cm"
    )
    print(
        f"Hip   : {max_jumps['hip']:.2f} cm"
    )

    print()

    if jump_records:
        print("RESULT: FAIL")
        print("Jumps greater than 3 cm were found:")

        for record in jump_records:
            print(
                f"- {record['measurement']} | "
                f"{record['weight_from']:.2f} -> "
                f"{record['weight_to']:.2f} | "
                f"{record['jump']:.2f} cm"
            )
    else:
        print("RESULT: PASS")
        print(
            "No measurement jump greater than "
            "3 cm was detected."
        )


def shape_effect_test(model, measurements):
    print()
    print("================================")
    print("MUSCLE EFFECT TEST")
    print("================================")
    print()

    for muscle in [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ]:
        result = measure_body(
            model,
            measurements,
            weight=0.5,
            muscle=muscle,
            proportions=0.5,
        )

        print(
            f"muscle={muscle:.2f} | "
            f"waist={result['waist']:.2f} cm | "
            f"chest={result['chest']:.2f} cm | "
            f"hip={result['hip']:.2f} cm"
        )

    print()
    print("================================")
    print("PROPORTIONS EFFECT TEST")
    print("================================")
    print()

    for proportions in [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ]:
        result = measure_body(
            model,
            measurements,
            weight=0.5,
            muscle=0.5,
            proportions=proportions,
        )

        print(
            f"proportions={proportions:.2f} | "
            f"waist={result['waist']:.2f} cm | "
            f"chest={result['chest']:.2f} cm | "
            f"hip={result['hip']:.2f} cm"
        )


def main():
    print("Starting Stage A...")

    model = anny.Anny()

    measurements = BodyMeasurements(
        model
    )

    continuity_test(
        model,
        measurements,
    )

    shape_effect_test(
        model,
        measurements,
    )

    print()
    print("================================")
    print("STAGE A COMPLETE")
    print("================================")


if __name__ == "__main__":
    main()