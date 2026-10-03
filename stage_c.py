import os
import numpy as np
import torch
import trimesh
import anny


# ============================================================
# STAGE C - EXPORT FINAL AVATAR
# ============================================================

OUTPUT_FILE = "avatar.glb"

# Final phenotype from Stage B
PHENOTYPE = {
    "gender": 1.0,       # female
    "age": 0.500000,
    "height": 0.571431,
    "weight": 0.882881,
    "muscle": 0.825434,
    "proportions": 0.999000,
}


def make_phenotype(model):
    phenotype = {
        key: 0.5
        for key in model.phenotype_labels
    }

    for key, value in PHENOTYPE.items():
        phenotype[key] = torch.tensor(
            [value],
            dtype=torch.float64
        )

    return phenotype


def main():

    print("=" * 60)
    print("STAGE C - EXPORT FINAL AVATAR")
    print("=" * 60)

    print("\nLoading Anny model...")

    model = anny.Anny().to(dtype=torch.float64)

    phenotype = make_phenotype(model)

    print("Generating final avatar...")

    output = model(
        phenotype_kwargs=phenotype
    )

    # --------------------------------------------------------
    # Get vertices and faces
    # --------------------------------------------------------

    vertices = (
        output["vertices"]
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    faces = (
        model.faces
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64)
    )

    print(f"\nOriginal vertices: {vertices.shape}")
    print(f"Original faces:    {faces.shape}")

    # --------------------------------------------------------
    # Create mesh
    # --------------------------------------------------------

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False
    )

    # --------------------------------------------------------
    # Anny uses Z-up.
    # Three.js / glTF will be prepared as Y-up.
    #
    # Rotate around X by -90 degrees:
    # Z-up -> Y-up
    # --------------------------------------------------------

    rotation = trimesh.transformations.rotation_matrix(
        -np.pi / 2,
        [1, 0, 0]
    )

    mesh.apply_transform(rotation)

    # --------------------------------------------------------
    # Put feet on ground: Y = 0
    # --------------------------------------------------------

    min_y = float(mesh.vertices[:, 1].min())

    mesh.vertices[:, 1] -= min_y

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    min_coords = mesh.vertices.min(axis=0)
    max_coords = mesh.vertices.max(axis=0)

    print("\nAfter Y-up conversion:")
    print("Minimum XYZ:", min_coords)
    print("Maximum XYZ:", max_coords)

    print(f"\nGround level Y: {mesh.vertices[:, 1].min():.6f} m")
    print(f"Maximum Y:     {mesh.vertices[:, 1].max():.6f} m")
    print(f"Height:        {mesh.vertices[:, 1].max() - mesh.vertices[:, 1].min():.6f} m")

    # --------------------------------------------------------
    # Export GLB
    # --------------------------------------------------------

    output_path = os.path.abspath(OUTPUT_FILE)

    mesh.export(output_path)

    print("\n" + "=" * 60)
    print("STAGE C COMPLETE")
    print("=" * 60)

    print(f"\nGLB created successfully:")
    print(output_path)

    print(f"\nFile size: {os.path.getsize(output_path) / 1024:.2f} KB")

    # --------------------------------------------------------
    # Reload exported GLB to make sure it is valid
    # --------------------------------------------------------

    print("\nValidating exported GLB...")

    loaded = trimesh.load(
        output_path,
        force="mesh"
    )

    print("Reloaded vertices:", loaded.vertices.shape)
    print("Reloaded faces:", loaded.faces.shape)

    print("\nRESULT: PASS")
    print("Avatar is ready for Three.js.")


if __name__ == "__main__":
    main()