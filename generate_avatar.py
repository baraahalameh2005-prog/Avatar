import argparse
import json
import numpy as np
import torch
import trimesh
import anny

AGE = 0.5
DT = torch.float64


def build(gender, height_cm, weight_kg, model, anth):
    def run(h, w):
        ph = {
            k: torch.full((1,), 0.5, dtype=DT)
            for k in model.phenotype_labels
        }

        ph["gender"] = torch.full((1,), gender, dtype=DT)
        ph["age"] = torch.full((1,), AGE, dtype=DT)
        ph["height"] = torch.full((1,), h, dtype=DT)
        ph["weight"] = torch.full((1,), w, dtype=DT)

        out = model(phenotype_kwargs=ph)

        m = {
            k: v.item()
            for k, v in anth(out["rest_vertices"]).items()
        }

        return out, m

    def solve(f, target):
        lo, hi = 0.0, 1.0

        for _ in range(30):
            mid = (lo + hi) / 2

            if f(mid) < target:
                lo = mid
            else:
                hi = mid

        return (lo + hi) / 2

    # حل الطول أولاً
    h = solve(
        lambda x: run(x, 0.5)[1]["height"],
        height_cm / 100
    )

    # ثم حل الوزن
    w = solve(
        lambda x: run(h, x)[1]["mass"],
        weight_kg
    )

    out, m = run(h, w)

    return out, m, h, w


def export(out, model, path):
    v = out["rest_vertices"][0].detach().cpu().numpy()

    f = model.faces.detach().cpu().numpy()

    # تحويل المحاور إلى نظام مناسب للعرض
    v = np.stack(
        [v[:, 0], v[:, 2], -v[:, 1]],
        axis=1
    )

    # وضع القدمين على الأرض
    v[:, 1] -= v[:, 1].min()

    mesh = trimesh.Trimesh(
        v,
        f,
        process=False
    )

    mesh.export(path)


if __name__ == "__main__":

    p = argparse.ArgumentParser()

    p.add_argument(
        "--gender",
        choices=["male", "female"],
        default="female"
    )

    p.add_argument(
        "--height",
        type=float,
        default=157
    )

    p.add_argument(
        "--weight",
        type=float,
        default=45
    )

    p.add_argument(
        "--out",
        default="avatar.glb"
    )

    a = p.parse_args()

    print("Loading Anny...")

    model = anny.Anny()

    anth = anny.Anthropometry(model)

    # Anny:
    # 0 = male
    # 1 = female
    g = 1.0 if a.gender == "female" else 0.0

    print("Solving body parameters...")

    out, m, h, w = build(
        g,
        a.height,
        a.weight,
        model,
        anth
    )

    print("Exporting avatar...")

    export(
        out,
        model,
        a.out
    )

    result = {
        "height_param": round(h, 3),
        "weight_param": round(w, 3),
        **{
            k: round(v, 3)
            for k, v in m.items()
        }
    }

    print()
    print(json.dumps(result, indent=2))

    if min(h, w) < 0.01 or max(h, w) > 0.99:
        print()
        print(
            "WARNING: target is at the edge of the "
            "model range, result is approximate"
        )

    print()
    print("Done.")