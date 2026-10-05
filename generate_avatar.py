import argparse
import json

import numpy as np
import torch
import trimesh
import anny


DT = torch.float64


def make_phenotype(
    model,
    gender,
    height,
    weight,
    age,
    muscle,
    proportions,
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

    phenotype["age"] = torch.full(
        (1,),
        age,
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


def build(
    model,
    gender,
    height,
    weight,
    age,
    muscle,
    proportions,
):
    phenotype = make_phenotype(
        model=model,
        gender=gender,
        height=height,
        weight=weight,
        age=age,
        muscle=muscle,
        proportions=proportions,
    )

    output = model(
        phenotype_kwargs=phenotype
    )

    return output


def export_avatar(
    output,
    model,
    path,
):
    vertices = (
        output["rest_vertices"][0]
        .detach()
        .cpu()
        .numpy()
    )

    faces = (
        model.faces
        .detach()
        .cpu()
        .numpy()
    )

    # Convert Anny's coordinate system
    # to the coordinate system used by the viewer.
    vertices = np.stack(
        [
            vertices[:, 0],
            vertices[:, 2],
            -vertices[:, 1],
        ],
        axis=1,
    )

    # Put the feet on the ground.
    vertices[:, 1] -= vertices[:, 1].min()

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
    )

    mesh.export(path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gender",
        choices=["male", "female"],
        default="female",
    )

    parser.add_argument(
        "--height-param",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--weight-param",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--age",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--muscle",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--proportions",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--out",
        default="avatar.glb",
    )

    args = parser.parse_args()

    print("Loading Anny...")

    model = anny.Anny()

    gender = (
        1.0
        if args.gender == "female"
        else 0.0
    )

    print("Building personalized avatar...")

    output = build(
        model=model,
        gender=gender,
        height=args.height_param,
        weight=args.weight_param,
        age=args.age,
        muscle=args.muscle,
        proportions=args.proportions,
    )

    print("Exporting avatar...")

    export_avatar(
        output=output,
        model=model,
        path=args.out,
    )

    result = {
        "gender": args.gender,
        "height_param": round(
            args.height_param,
            4,
        ),
        "weight_param": round(
            args.weight_param,
            4,
        ),
        "age": round(
            args.age,
            4,
        ),
        "muscle": round(
            args.muscle,
            4,
        ),
        "proportions": round(
            args.proportions,
            4,
        ),
        "output": args.out,
    }

    print()
    print("AVATAR GENERATED")
    print("----------------")
    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()