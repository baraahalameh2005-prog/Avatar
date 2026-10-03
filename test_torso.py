import torch
import anny
import trimesh
import numpy as np


DT = torch.float64


def make_model(model, gender, weight):
    ph = {
        k: torch.full((1,), 0.5, dtype=DT)
        for k in model.phenotype_labels
    }

    ph["gender"] = torch.full((1,), gender, dtype=DT)
    ph["weight"] = torch.full((1,), weight, dtype=DT)

    return model(phenotype_kwargs=ph)


def perimeter(points):
    closed = np.vstack([points, points[0]])

    return float(
        np.linalg.norm(
            np.diff(closed, axis=0),
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


def torso_loops(output, model, z):
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

    results = []

    for points in section.discrete:

        if len(points) < 6:
            continue

        cx = float(points[:, 0].mean())

        results.append({
            "center_x": cx,
            "area": area(points),
            "circumference": perimeter(points),
            "width": float(
                points[:, 0].max()
                - points[:, 0].min()
            ),
            "depth": float(
                points[:, 1].max()
                - points[:, 1].min()
            ),
        })

    return results


model = anny.Anny()


tests = [
    ("Female weight 0", 1.0, 0.0),
    ("Female weight 1", 1.0, 1.0),
    ("Male weight 0", 0.0, 0.0),
    ("Male weight 1", 0.0, 1.0),
]


for name, gender, weight in tests:

    output = make_model(
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
        anny.Anthropometry(model)
        .waist_vertex_indices
    )

    waist_z = float(
        vertices[waist_indices, 2].mean()
    )

    height = float(
        vertices[:, 2].max()
        - vertices[:, 2].min()
    )

    print()
    print("=" * 50)
    print(name)
    print(
        f"Height : {height * 100:.2f} cm"
    )
    print(
        f"Waist Z: {waist_z:.4f}"
    )

    chest_z = 0.386

    print()
    print(
        f"CHEST Z = {chest_z:.4f}"
    )

    loops = torso_loops(
        output,
        model,
        chest_z,
    )

    loops.sort(
        key=lambda x: x["area"],
        reverse=True,
    )

    for i, r in enumerate(loops[:5]):

        print(
            f"Loop {i+1}: "
            f"centerX={r['center_x']:.4f} "
            f"area={r['area']:.4f} "
            f"circ={r['circumference']*100:.2f} cm "
            f"width={r['width']*100:.2f} cm "
            f"depth={r['depth']*100:.2f} cm"
        )