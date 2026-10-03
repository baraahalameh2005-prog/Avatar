import torch
import anny
import trimesh
import numpy as np


DT = torch.float64


def build(model, gender, weight):

    ph = {
        k: torch.full((1,), 0.5, dtype=DT)
        for k in model.phenotype_labels
    }

    ph["gender"] = torch.full((1,), gender, dtype=DT)
    ph["weight"] = torch.full((1,), weight, dtype=DT)

    return model(
        phenotype_kwargs=ph
    )


def perimeter(points):

    closed = np.vstack([
        points,
        points[0],
    ])

    return float(
        np.linalg.norm(
            np.diff(
                closed,
                axis=0,
            ),
            axis=1,
        ).sum()
    )


def area(points):

    x = points[:, 0]
    y = points[:, 1]

    return float(
        0.5 * abs(
            np.sum(
                x * np.roll(y, -1)
                - np.roll(x, -1) * y
            )
        )
    )


def chest_candidates(
    output,
    model,
    z,
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

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
    )

    section = mesh.section(
        plane_origin=[0, 0, z],
        plane_normal=[0, 0, 1],
    )

    if section is None:
        return []

    result = []

    for points in section.discrete:

        if len(points) < 6:
            continue

        cx = abs(float(points[:, 0].mean()))

        if cx > 0.05:
            continue

        result.append({
            "area": area(points),
            "circ": perimeter(points),
            "width": (
                points[:, 0].max()
                - points[:, 0].min()
            ),
            "depth": (
                points[:, 1].max()
                - points[:, 1].min()
            ),
        })

    return sorted(
        result,
        key=lambda x: x["area"],
        reverse=True,
    )


model = anny.Anny()
anth = anny.Anthropometry(model)


tests = [
    ("Female weight 0", 1.0, 0.0),
    ("Female weight 1", 1.0, 1.0),
    ("Male weight 0", 0.0, 0.0),
    ("Male weight 1", 0.0, 1.0),
]


for name, gender, weight in tests:

    output = build(
        model,
        gender,
        weight,
    )

    vertices = (
        output["rest_vertices"][0]
        .detach()
        .cpu()
        .numpy()
    )

    waist_indices = (
        anth.waist_vertex_indices
    )

    waist_z = float(
        vertices[
            waist_indices,
            2,
        ].mean()
    )

    height = float(
        vertices[:, 2].max()
        - vertices[:, 2].min()
    )

    print()
    print("=" * 70)
    print(name)

    print(
        f"Height = {height * 100:.2f} cm"
    )

    print(
        f"Waist Z = {waist_z:.4f}"
    )

    print()
    print(
        "offset | z       | circ    | width   | depth"
    )
    print("-" * 55)

    for offset in np.arange(
        0.120,
        0.151,
        0.002,
    ):

        z = waist_z + height * offset

        candidates = chest_candidates(
            output,
            model,
            z,
        )

        if not candidates:
            continue

        best = candidates[0]

        print(
            f"{offset:.3f} | "
            f"{z:.4f} | "
            f"{best['circ']*100:7.2f} | "
            f"{best['width']*100:7.2f} | "
            f"{best['depth']*100:7.2f}"
        )