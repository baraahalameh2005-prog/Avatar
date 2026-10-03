import torch
import anny
import trimesh
import numpy as np


DT = torch.float64

model = anny.Anny()
anth = anny.Anthropometry(model)

ph = {
    k: torch.full((1,), 0.5, dtype=DT)
    for k in model.phenotype_labels
}

ph["gender"] = torch.full((1,), 1.0, dtype=DT)
ph["age"] = torch.full((1,), 0.5, dtype=DT)
ph["weight"] = torch.full((1,), 0.5, dtype=DT)

output = model(
    phenotype_kwargs=ph
)

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

waist_indices = anth.waist_vertex_indices

waist_z = float(
    vertices[waist_indices, 2].mean()
)

height = float(
    vertices[:, 2].max()
    - vertices[:, 2].min()
)


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


print()
print("DEFAULT FEMALE")
print("----------------")
print(f"Height : {height * 100:.2f} cm")
print(f"Waist Z: {waist_z:.4f}")

print()
print("offset | z       | circumference | width | depth")
print("-" * 60)


for offset in np.arange(0.10, 0.201, 0.001):

    z = waist_z + height * offset

    section = mesh.section(
        plane_origin=[0, 0, z],
        plane_normal=[0, 0, 1],
    )

    if section is None:
        continue

    candidates = []

    for points in section.discrete:

        if len(points) < 6:
            continue

        if abs(float(points[:, 0].mean())) > 0.05:
            continue

        candidates.append({
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

    if not candidates:
        continue

    best = max(
        candidates,
        key=lambda x: x["area"],
    )

    print(
        f"{offset:.3f} | "
        f"{z:.4f} | "
        f"{best['circ'] * 100:7.2f} cm | "
        f"{best['width'] * 100:5.2f} | "
        f"{best['depth'] * 100:5.2f}"
    )