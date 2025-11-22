import numpy as np
import json


def interpolate(p1, p2):
    return (p1 + p2) / 2.0


def convert_landmarks_98_to_106(lm98):
    lm = np.array(lm98).reshape(-1, 2)
    new = []

    # Lips midpoints
    new.append(interpolate(lm[52], lm[61]))
    new.append(interpolate(lm[66], lm[70]))

    # Eye 4 midpoints
    new.append(interpolate(lm[72], lm[73]))
    new.append(interpolate(lm[75], lm[76]))
    new.append(interpolate(lm[68], lm[69]))
    new.append(interpolate(lm[71], lm[70]))

    # Chin midpoint
    new.append(interpolate(lm[6], lm[10]))

    # Nose bridge midpoint
    new.append(interpolate(lm[27], lm[28]))

    return np.vstack([lm, np.array(new)])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--output")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        ann = json.load(f)

    out = {}
    for k, v in ann.items():
        out[k] = {
            "landmarks_106": convert_landmarks_98_to_106(v["landmarks"]).tolist(),
            "bbox": v["bbox"]
        }

    with open(args.output, "w") as f:
        json.dump(out, f, indent=4)