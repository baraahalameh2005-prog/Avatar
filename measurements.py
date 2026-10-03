import numpy as np
import anny
import trimesh


class BodyMeasurements:
    def __init__(self, model):
        self.model = model
        self.anthropometry = anny.Anthropometry(model)

    def _mesh(self, rest_vertices):
        vertices = rest_vertices[0].detach().cpu().numpy()
        faces = self.model.faces.detach().cpu().numpy()

        return trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            process=False,
        )

    def _perimeter(self, points):
        closed = np.vstack([points, points[0]])

        return float(
            np.linalg.norm(
                np.diff(closed, axis=0),
                axis=1,
            ).sum()
        )

    def _loop_area(self, points):
        return float(
            0.5
            * abs(
                np.dot(
                    points[:, 0],
                    np.roll(points[:, 1], -1),
                )
                - np.dot(
                    points[:, 1],
                    np.roll(points[:, 0], -1),
                )
            )
        )

    def _section(self, rest_vertices, origin, normal):
        mesh = self._mesh(rest_vertices)

        origin = np.asarray(origin, dtype=float)
        normal = np.asarray(normal, dtype=float)

        length = np.linalg.norm(normal)

        if length < 1e-8:
            return None

        normal = normal / length

        section = mesh.section(
            plane_origin=origin,
            plane_normal=normal,
        )

        if section is None:
            return None

        candidates = []

        for points in section.discrete:
            if len(points) < 3:
                continue

            perimeter = self._perimeter(points)
            area = self._loop_area(points)

            width = (
                points[:, 0].max()
                - points[:, 0].min()
            )

            candidates.append(
                {
                    "area": area,
                    "circumference": perimeter,
                    "width": float(width),
                    "points": points,
                }
            )

        if not candidates:
            return None

        best = max(
            candidates,
            key=lambda x: x["area"],
        )

        return {
            "circumference": best["circumference"],
            "width": best["width"],
            "points": best["points"],
        }

    def _horizontal_section(self, rest_vertices, z):
        return self._section(
            rest_vertices,
            [0.0, 0.0, z],
            [0.0, 0.0, 1.0],
        )

    def _waist_z(self, rest_vertices):
        vertices = (
            rest_vertices[0]
            .detach()
            .cpu()
            .numpy()
        )

        indices = self.anthropometry.waist_vertex_indices

        return float(
            vertices[indices, 2].mean()
        )

    def _chest_section(self, rest_vertices):
        """
        Stable chest measurement.

        The chest is measured around a narrow anatomical
        band above the waist instead of selecting the largest
        circumference from a wide scan.

        This prevents the selected chest region from jumping
        between different torso loops as body weight changes.
        """

        base = self.anthropometry(rest_vertices)

        height = float(
            base["height"].item()
        )

        waist_z = self._waist_z(
            rest_vertices
        )

        target_offset = 0.145

        target_z = (
            waist_z
            + height * target_offset
        )

        mesh = self._mesh(rest_vertices)

        candidates = []

        for offset in np.arange(
            0.135,
            0.151,
            0.001,
        ):
            z = (
                waist_z
                + height * offset
            )

            section = mesh.section(
                plane_origin=[0.0, 0.0, z],
                plane_normal=[0.0, 0.0, 1.0],
            )

            if section is None:
                continue

            loops = []

            for points in section.discrete:

                if len(points) < 6:
                    continue

                if abs(
                    float(points[:, 0].mean())
                ) > 0.05:
                    continue

                perimeter = self._perimeter(
                    points
                )

                area = self._loop_area(
                    points
                )

                width = (
                    points[:, 0].max()
                    - points[:, 0].min()
                )

                # Reject oversized torso loops.
                if width > 0.45:
                    continue

                loops.append(
                    {
                        "z": float(z),
                        "circumference": perimeter,
                        "width": float(width),
                        "area": area,
                    }
                )

            if not loops:
                continue

            best_loop = max(
                loops,
                key=lambda x: x["area"],
            )

            best_loop["distance"] = abs(
                best_loop["z"] - target_z
            )

            candidates.append(
                best_loop
            )

        if not candidates:
            return None

        candidates.sort(
            key=lambda x: (
                x["distance"],
                -x["area"],
            )
        )

        best = candidates[0]

        return {
            "z": best["z"],
            "circumference": best["circumference"],
            "width": best["width"],
            "points": None,
        }

    def _hip_section(self, rest_vertices):
        base = self.anthropometry(rest_vertices)

        height = float(
            base["height"].item()
        )

        waist_z = self._waist_z(
            rest_vertices
        )

        hip_z = waist_z - height * 0.10

        mesh = self._mesh(rest_vertices)

        section = mesh.section(
            plane_origin=[0.0, 0.0, hip_z],
            plane_normal=[0.0, 0.0, 1.0],
        )

        if section is None:
            return None

        candidates = []

        for points in section.discrete:
            if len(points) < 20:
                continue

            if abs(points[:, 0].mean()) >= 0.1:
                continue

            perimeter = self._perimeter(points)

            width = (
                points[:, 0].max()
                - points[:, 0].min()
            )

            candidates.append(
                {
                    "circumference": perimeter,
                    "width": float(width),
                    "points": points,
                }
            )

        if not candidates:
            return None

        best = max(
            candidates,
            key=lambda x: x["circumference"],
        )

        return {
            "z": float(hip_z),
            "circumference": best["circumference"],
            "width": best["width"],
        }

    def _bone_origin(self, rest_bone_poses, index):
        pose = (
            rest_bone_poses[0, index]
            .detach()
            .cpu()
            .numpy()
        )

        return pose[:3, 3]

    def _bone_axis(
        self,
        rest_bone_poses,
        start_index,
        end_index,
    ):
        start = self._bone_origin(
            rest_bone_poses,
            start_index,
        )

        end = self._bone_origin(
            rest_bone_poses,
            end_index,
        )

        axis = end - start

        length = np.linalg.norm(axis)

        if length < 1e-8:
            return None

        return axis / length

    def _limb_sections(
        self,
        rest_vertices,
        rest_bone_poses,
        start_index,
        end_index,
        t,
    ):
        start = self._bone_origin(
            rest_bone_poses,
            start_index,
        )

        end = self._bone_origin(
            rest_bone_poses,
            end_index,
        )

        origin = start + (end - start) * t

        axis = self._bone_axis(
            rest_bone_poses,
            start_index,
            end_index,
        )

        if axis is None:
            return []

        mesh = self._mesh(rest_vertices)

        section = mesh.section(
            plane_origin=origin,
            plane_normal=axis,
        )

        if section is None:
            return []

        results = []

        for points in section.discrete:
            if len(points) < 10:
                continue

            perimeter = self._perimeter(points)

            results.append(
                {
                    "circumference": perimeter,
                    "points": points,
                }
            )

        return results

    def _smallest_limb_section(
        self,
        rest_vertices,
        rest_bone_poses,
        start_index,
        end_index,
        t,
    ):
        sections = self._limb_sections(
            rest_vertices,
            rest_bone_poses,
            start_index,
            end_index,
            t,
        )

        if not sections:
            return None

        return min(
            sections,
            key=lambda x: x["circumference"],
        )

    def _leg_sections(
        self,
        rest_vertices,
        rest_bone_poses,
        start_index,
        end_index,
        t,
    ):
        sections = self._limb_sections(
            rest_vertices,
            rest_bone_poses,
            start_index,
            end_index,
            t,
        )

        if len(sections) < 2:
            return sections

        sections.sort(
            key=lambda x: x["circumference"]
        )

        return sections[:2]

    def measure(self, output):
        rest_vertices = output["rest_vertices"]
        rest_bone_poses = output["rest_bone_poses"]

        base = self.anthropometry(
            rest_vertices
        )

        height = float(
            base["height"].item()
        )

        mass = float(
            base["mass"].item()
        )

        bmi = float(
            base["bmi"].item()
        )

        vertices = (
            rest_vertices[0]
            .detach()
            .cpu()
            .numpy()
        )

        waist_indices = (
            self.anthropometry.waist_vertex_indices
        )

        waist_width = (
            vertices[waist_indices, 0].max()
            - vertices[waist_indices, 0].min()
        )

        chest = self._chest_section(
            rest_vertices
        )

        hip = self._hip_section(
            rest_vertices
        )

        neck = self._horizontal_section(
            rest_vertices,
            0.580,
        )

        upper_arm = self._smallest_limb_section(
            rest_vertices,
            rest_bone_poses,
            48,
            49,
            0.70,
        )

        forearm = self._smallest_limb_section(
            rest_vertices,
            rest_bone_poses,
            50,
            51,
            0.45,
        )

        thigh_sections = self._leg_sections(
            rest_vertices,
            rest_bone_poses,
            3,
            4,
            0.50,
        )

        calf_sections = self._leg_sections(
            rest_vertices,
            rest_bone_poses,
            4,
            5,
            0.50,
        )

        thigh_left = (
            thigh_sections[0]["circumference"]
            if len(thigh_sections) > 0
            else None
        )

        thigh_right = (
            thigh_sections[1]["circumference"]
            if len(thigh_sections) > 1
            else None
        )

        calf_left = (
            calf_sections[0]["circumference"]
            if len(calf_sections) > 0
            else None
        )

        calf_right = (
            calf_sections[1]["circumference"]
            if len(calf_sections) > 1
            else None
        )

        thigh_values = [
            x
            for x in [
                thigh_left,
                thigh_right,
            ]
            if x is not None
        ]

        calf_values = [
            x
            for x in [
                calf_left,
                calf_right,
            ]
            if x is not None
        ]

        return {
            "height_m": height,
            "mass_kg": mass,
            "bmi": bmi,

            "waist_m": float(
                base["waist_circumference"].item()
            ),

            "waist_width_m": float(
                waist_width
            ),

            "chest_m": (
                chest["circumference"]
                if chest
                else None
            ),

            "chest_width_m": (
                chest["width"]
                if chest
                else None
            ),

            "hip_m": (
                hip["circumference"]
                if hip
                else None
            ),

            "hip_width_m": (
                hip["width"]
                if hip
                else None
            ),

            "hip_z": (
                hip["z"]
                if hip
                else None
            ),

            "neck_m": (
                neck["circumference"]
                if neck
                else None
            ),

            "upper_arm_m": (
                upper_arm["circumference"]
                if upper_arm
                else None
            ),

            "forearm_m": (
                forearm["circumference"]
                if forearm
                else None
            ),

            "thigh_m": (
                float(np.mean(thigh_values))
                if thigh_values
                else None
            ),

            "thigh_left_m": thigh_left,

            "thigh_right_m": thigh_right,

            "calf_m": (
                float(np.mean(calf_values))
                if calf_values
                else None
            ),

            "calf_left_m": calf_left,

            "calf_right_m": calf_right,
        }


if __name__ == "__main__":
    model = anny.Anny()

    measurements = BodyMeasurements(
        model
    )

    output = model()

    result = measurements.measure(
        output
    )

    print()
    print("BODY MEASUREMENTS")
    print("-----------------")

    print(
        f"Height       : "
        f"{result['height_m'] * 100:.2f} cm"
    )

    print(
        f"Weight       : "
        f"{result['mass_kg']:.2f} kg"
    )

    print(
        f"BMI          : "
        f"{result['bmi']:.2f}"
    )

    print()

    print(
        f"Waist        : "
        f"{result['waist_m'] * 100:.2f} cm"
    )

    print(
        f"Waist width  : "
        f"{result['waist_width_m'] * 100:.2f} cm"
    )

    print()

    print(
        f"Chest        : "
        f"{result['chest_m'] * 100:.2f} cm"
        if result["chest_m"] is not None
        else "Chest        : N/A"
    )

    print(
        f"Chest width  : "
        f"{result['chest_width_m'] * 100:.2f} cm"
        if result["chest_width_m"] is not None
        else "Chest width  : N/A"
    )

    print()

    print(
        f"Hip          : "
        f"{result['hip_m'] * 100:.2f} cm"
        if result["hip_m"] is not None
        else "Hip          : N/A"
    )

    print(
        f"Hip width    : "
        f"{result['hip_width_m'] * 100:.2f} cm"
        if result["hip_width_m"] is not None
        else "Hip width    : N/A"
    )

    print()

    print(
        f"Neck         : "
        f"{result['neck_m'] * 100:.2f} cm"
        if result["neck_m"] is not None
        else "Neck         : N/A"
    )

    print(
        f"Upper arm    : "
        f"{result['upper_arm_m'] * 100:.2f} cm"
        if result["upper_arm_m"] is not None
        else "Upper arm    : N/A"
    )

    print(
        f"Forearm      : "
        f"{result['forearm_m'] * 100:.2f} cm"
        if result["forearm_m"] is not None
        else "Forearm      : N/A"
    )

    print()

    print(
        f"Thigh        : "
        f"{result['thigh_m'] * 100:.2f} cm"
        if result["thigh_m"] is not None
        else "Thigh        : N/A"
    )

    print(
        f"  Left       : "
        f"{result['thigh_left_m'] * 100:.2f} cm"
        if result["thigh_left_m"] is not None
        else "  Left       : N/A"
    )

    print(
        f"  Right      : "
        f"{result['thigh_right_m'] * 100:.2f} cm"
        if result["thigh_right_m"] is not None
        else "  Right      : N/A"
    )

    print()

    print(
        f"Calf         : "
        f"{result['calf_m'] * 100:.2f} cm"
        if result["calf_m"] is not None
        else "Calf         : N/A"
    )

    print(
        f"  Left       : "
        f"{result['calf_left_m'] * 100:.2f} cm"
        if result["calf_left_m"] is not None
        else "  Left       : N/A"
    )

    print(
        f"  Right      : "
        f"{result['calf_right_m'] * 100:.2f} cm"
        if result["calf_right_m"] is not None
        else "  Right      : N/A"
    )
