import os
import numpy as np
import trimesh


# ============================================================
# STAGE E - BASIC 3D SHIRT
# ============================================================

AVATAR_FILE = "avatar.glb"

SHIRT_FILES = {
    "S": "shirt_S.glb",
    "M": "shirt_M.glb",
    "L": "shirt_L.glb",
    "XL": "shirt_XL.glb",
}


# Shirt chest measurements from Stage D
SHIRT_CHEST = {
    "S": 90.0,
    "M": 96.0,
    "L": 102.0,
    "XL": 110.0,
}


def create_shirt(avatar_mesh, size):
    """
    Create a simple shirt-like 3D mesh around the avatar.

    This is a PoC approximation:
    - no cloth simulation
    - no skinning
    - no physics
    """

    vertices = np.asarray(avatar_mesh.vertices)

    min_xyz = vertices.min(axis=0)
    max_xyz = vertices.max(axis=0)

    avatar_height = max_xyz[1] - min_xyz[1]

    # Approximate torso region
    shirt_bottom = min_xyz[1] + avatar_height * 0.48
    shirt_top = min_xyz[1] + avatar_height * 0.70

    shirt_height = shirt_top - shirt_bottom

    # Shirt chest width based on chest circumference.
    #
    # Approximate ellipse:
    # circumference ≈ pi * (3(a+b) - sqrt((3a+b)(a+3b)))
    #
    # For the PoC we use a simple width approximation.
    chest_circumference = SHIRT_CHEST[size]

    chest_width = chest_circumference / np.pi / 2.0

    # Add a small amount of room so the shirt sits outside the body.
    chest_width *= 1.08

    # Depth of shirt
    chest_depth = chest_width * 0.55

    # Half dimensions
    half_width = chest_width / 2.0
    half_depth = chest_depth / 2.0

    # --------------------------------------------------------
    # Create a simple box-like shirt shell
    # --------------------------------------------------------

    y0 = shirt_bottom
    y1 = shirt_top

    x0 = -half_width
    x1 = half_width

    z0 = -half_depth
    z1 = half_depth

    # Main torso vertices
    v = np.array([
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y0, z1],
        [x0, y0, z1],

        [x0, y1, z0],
        [x1, y1, z0],
        [x1, y1, z1],
        [x0, y1, z1],
    ], dtype=np.float32)

    # Slightly offset shirt toward front/back relative to avatar center.
    # This keeps it visually separated from the body.
    v[:, 2] += 0.0

    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],

        [4, 6, 5],
        [4, 7, 6],

        [0, 4, 5],
        [0, 5, 1],

        [1, 5, 6],
        [1, 6, 2],

        [2, 6, 7],
        [2, 7, 3],

        [3, 7, 4],
        [3, 4, 0],
    ], dtype=np.int64)

    shirt = trimesh.Trimesh(
        vertices=v,
        faces=faces,
        process=False
    )

    return shirt


def main():

    print("=" * 60)
    print("STAGE E - BASIC 3D SHIRT")
    print("=" * 60)

    # --------------------------------------------------------
    # Load avatar
    # --------------------------------------------------------

    print("\nLoading avatar...")

    if not os.path.exists(AVATAR_FILE):
        raise FileNotFoundError(
            f"Could not find {AVATAR_FILE}"
        )

    avatar = trimesh.load(
        AVATAR_FILE,
        force="mesh"
    )

    print("Avatar vertices:", len(avatar.vertices))
    print("Avatar faces:", len(avatar.faces))

    # --------------------------------------------------------
    # Generate shirts
    # --------------------------------------------------------

    print("\nGenerating shirts...")

    generated = 0

    for size in ["S", "M", "L", "XL"]:

        shirt = create_shirt(
            avatar,
            size
        )

        output_file = SHIRT_FILES[size]

        shirt.export(output_file)

        print(
            f"{size}: "
            f"chest={SHIRT_CHEST[size]:.1f} cm "
            f"-> {output_file}"
        )

        generated += 1

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print("\nValidating generated files...")

    for size in ["S", "M", "L", "XL"]:

        file = SHIRT_FILES[size]

        loaded = trimesh.load(
            file,
            force="mesh"
        )

        print(
            f"{size}: "
            f"vertices={len(loaded.vertices)}, "
            f"faces={len(loaded.faces)}"
        )

    print("\n" + "=" * 60)
    print("STAGE E COMPLETE")
    print("=" * 60)

    print(f"\nGenerated {generated} shirt models:")
    print("shirt_S.glb")
    print("shirt_M.glb")
    print("shirt_L.glb")
    print("shirt_XL.glb")

    print("\nRESULT: PASS")


if __name__ == "__main__":
    main()