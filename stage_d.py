# ============================================================
# STAGE D - GARMENT FIT LOGIC
# ============================================================

# User chest measurement from Stage B / final avatar
BODY_CHEST_CM = 87.15


# Garment chest measurements
GARMENT_SIZES = {
    "S": 90.0,
    "M": 96.0,
    "L": 102.0,
    "XL": 110.0,
}


# Fit thresholds
SNUG_MAX = 4.0
REGULAR_MAX = 12.0


def calculate_fit(body_chest, garment_chest):
    """
    Calculate ease and fit category.
    """

    ease = garment_chest - body_chest

    if ease < 0:
        label = "Too Small"
    elif ease <= SNUG_MAX:
        label = "Snug"
    elif ease <= REGULAR_MAX:
        label = "Regular"
    else:
        label = "Oversized"

    return ease, label


def find_recommended_size(body_chest):
    """
    Find the smallest garment size that is not too small.
    """

    for size, garment_chest in GARMENT_SIZES.items():

        ease, label = calculate_fit(
            body_chest,
            garment_chest
        )

        if ease >= 0:
            return size, garment_chest, ease, label

    # If even XL is too small
    size = "XL"
    garment_chest = GARMENT_SIZES[size]
    ease, label = calculate_fit(
        body_chest,
        garment_chest
    )

    return size, garment_chest, ease, label


def main():

    print("=" * 60)
    print("STAGE D - GARMENT FIT LOGIC")
    print("=" * 60)

    print(f"\nBody chest: {BODY_CHEST_CM:.2f} cm")

    print("\nSize analysis:")
    print("-" * 60)

    for size, garment_chest in GARMENT_SIZES.items():

        ease, label = calculate_fit(
            BODY_CHEST_CM,
            garment_chest
        )

        print(
            f"{size:>2} | "
            f"Garment chest: {garment_chest:6.2f} cm | "
            f"Ease: {ease:6.2f} cm | "
            f"Fit: {label}"
        )

    recommended_size, garment_chest, ease, label = \
        find_recommended_size(BODY_CHEST_CM)

    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)

    print(f"Recommended size: {recommended_size}")
    print(f"Garment chest:    {garment_chest:.2f} cm")
    print(f"Ease:             {ease:.2f} cm")
    print(f"Fit:              {label}")

    print("\n" + "=" * 60)
    print("STAGE D COMPLETE")
    print("=" * 60)

    print("\nRESULT: PASS")


if __name__ == "__main__":
    main()