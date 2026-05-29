"""
Copy N test-split .off files per class into assets/samples/{class}/.
Run once after training, before committing or deploying to HF Spaces.

    python scripts/prepare_samples.py --data ../datasets/ModelNet40 --n 5
"""
import argparse
import os
import random
import shutil


def prepare(data_root: str, out_root: str, n: int, seed: int):
    random.seed(seed)
    classes = sorted(
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d, "train"))
    )
    for cls in classes:
        src_dir = os.path.join(data_root, cls, "test")
        dst_dir = os.path.join(out_root, cls)
        if not os.path.isdir(src_dir):
            print(f"  skip {cls}: no test folder")
            continue
        files = [f for f in os.listdir(src_dir) if f.endswith(".off")]
        chosen = random.sample(files, min(n, len(files)))
        os.makedirs(dst_dir, exist_ok=True)
        for f in chosen:
            shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))
        print(f"  {cls}: {len(chosen)} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../datasets/ModelNet40")
    parser.add_argument("--out",  default="assets/samples")
    parser.add_argument("--n",    type=int, default=5, help="files per class")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(f"Copying {args.n} test files per class → {args.out}")
    prepare(args.data, args.out, args.n, args.seed)
    print("Done.")
