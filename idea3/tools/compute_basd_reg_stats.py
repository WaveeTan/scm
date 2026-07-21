"""Compute BASD-Reg boundaries from DIOR-R training XML annotations."""

import argparse
import math
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a quantile from an empty collection")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def object_geometry(obj: ET.Element) -> tuple[float, float, float]:
    box = obj.find("robndbox")
    if box is None:
        raise ValueError("DIOR-R object is missing robndbox")
    names = (
        ("x_left_top", "y_left_top"),
        ("x_right_top", "y_right_top"),
        ("x_right_bottom", "y_right_bottom"),
        ("x_left_bottom", "y_left_bottom"),
    )
    points = [(float(box.find(x).text), float(box.find(y).text)) for x, y in names]
    center_x = sum(point[0] for point in points) / 4
    center_y = sum(point[1] for point in points) / 4
    width = math.dist(points[0], points[1])
    height = math.dist(points[1], points[2])
    return center_x, center_y, abs(width * height)


def image_statistics(xml_path: Path, density_k: int) -> list[tuple[float, float]]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = float(size.find("width").text)
    height = float(size.find("height").text)
    geometries = [object_geometry(obj) for obj in root.findall("object")]
    centers = [(x / width, y / height) for x, y, _ in geometries]
    output = []
    for index, (_, _, area) in enumerate(geometries):
        scale = math.sqrt(area / (width * height))
        if len(centers) <= 1:
            density = 0.0
        else:
            k_eff = min(density_k, len(centers) - 1)
            distances = sorted(
                math.dist(centers[index], center)
                for other, center in enumerate(centers)
                if other != index
            )
            radius = distances[k_eff - 1]
            density = math.log1p(k_eff / (radius * radius + 1e-6))
        output.append((scale, density))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=("train", "val"))
    parser.add_argument("--density-k", type=int, default=3)
    args = parser.parse_args()
    if any(split.lower() == "test" for split in args.splits):
        raise ValueError("test split must never be used for BASD-Reg statistics")

    annotations = args.data_root / "Annotations" / "Oriented Bounding Boxes"
    records: list[tuple[float, float]] = []
    image_count = 0
    for split in args.splits:
        split_file = args.data_root / "ImageSets" / "Main" / f"{split}.txt"
        image_ids = [line.strip() for line in split_file.read_text().splitlines()]
        for image_id in image_ids:
            if not image_id:
                continue
            records.extend(
                image_statistics(annotations / f"{image_id}.xml", args.density_k)
            )
            image_count += 1

    scales = [scale for scale, _ in records]
    q25, q50, q75 = (quantile(scales, q) for q in (0.25, 0.50, 0.75))
    lower_scale_density = [density for scale, density in records if scale < q50]
    density_median = statistics.median(lower_scale_density)
    print(f"images={image_count}, boxes={len(records)}")
    print(f"scale_boundaries=({q25:.8f}, {q50:.8f}, {q75:.8f})")
    print(f"density_boundary={density_median:.8f}")


if __name__ == "__main__":
    main()
