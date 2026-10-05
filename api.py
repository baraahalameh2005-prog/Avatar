from pathlib import Path
import time

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anny

from fit_to_measurements import fit_body
from generate_avatar import build, export_avatar


app = FastAPI()


# ===============================
# CORS
# ===============================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===============================
# PATHS
# ===============================

BASE_DIR = Path(__file__).resolve().parent

AVATAR_PATH = BASE_DIR / "generated_avatar.glb"


# ===============================
# LOAD ANNY ONCE
# ===============================

print("Loading Anny...")

model = anny.Anny()

print("Anny loaded successfully!")


# ===============================
# MEASUREMENTS
# ===============================

from measurements import BodyMeasurements

measurements = BodyMeasurements(model)


# ===============================
# REQUEST MODEL
# ===============================

class AvatarRequest(BaseModel):
    gender: str = "female"
    height: float
    weight: float
    waist: float
    chest: float
    hip: float


# ===============================
# STATIC FILES
# ===============================

app.mount(
    "/models",
    StaticFiles(directory=BASE_DIR),
    name="models",
)

app.mount(
    "/test",
    StaticFiles(directory=BASE_DIR, html=True),
    name="test",
)


# ===============================
# ROOT
# ===============================

@app.get("/")
def root():
    return {
        "message": "Avatar API is running"
    }


# ===============================
# GENERATE AVATAR
# ===============================

@app.post("/generate-avatar")
def generate_avatar(data: AvatarRequest):

    print()
    print("Generating personalized avatar...")

    # ---------------------------
    # 1. Fit body measurements
    # ---------------------------

    start_time = time.perf_counter()

    fitted = fit_body(
        model=model,
        measurements=measurements,
        gender=data.gender,
        height_cm=data.height,
        weight_kg=data.weight,
        waist_cm=data.waist,
        chest_cm=data.chest,
        hip_cm=data.hip,
    )

    fit_time = time.perf_counter() - start_time

    print(f"Fit time: {fit_time:.2f} seconds")

    params = fitted["parameters"]

    # ---------------------------
    # 2. Convert gender
    # ---------------------------

    gender_value = (
        1.0
        if data.gender == "female"
        else 0.0
    )

    # ---------------------------
    # 3. Build avatar
    # ---------------------------

    output = build(
        model=model,
        gender=gender_value,
        height=params["height"],
        weight=params["weight"],
        age=params["age"],
        muscle=params["muscle"],
        proportions=params["proportions"],
    )

    # ---------------------------
    # 4. Export GLB
    # ---------------------------

    export_avatar(
        output=output,
        model=model,
        path=AVATAR_PATH,
    )

    print("Avatar generated successfully!")

    # ---------------------------
    # 5. Return response
    # ---------------------------

    return {
        "message": "Avatar generated successfully",

        "avatar_url": "/models/generated_avatar.glb",

        "parameters": params,

        "measurements": {
            key: float(value)
            for key, value in fitted["result"].items()
            if value is not None and key != "hip_z"
        },
    }