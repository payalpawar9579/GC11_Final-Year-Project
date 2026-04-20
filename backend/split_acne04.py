import random
import shutil
from pathlib import Path

# =========================
# PATHS
# =========================
# Change these if needed
SOURCE_ROOT = Path(r"..\acne_1024")   # folder containing acne0_1024, acne1_1024, ...
OUT_ROOT = Path(r"..\dataset")        # output dataset folder

# =========================
# SPLIT RATIOS
# =========================
TRAIN = 0.80
VAL = 0.10
TEST = 0.10

# =========================
# CLASS FOLDERS
# =========================
CLASS_FOLDERS = {
    "0": "acne0_1024",
    "1": "acne1_1024",
    "2": "acne2_1024",
    "3": "acne3_1024",
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def list_images(folder: Path):
    files = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return files

def ensure_dirs():
    for split in ["train", "val", "test"]:
        for cls in CLASS_FOLDERS.keys():
            (OUT_ROOT / split / cls).mkdir(parents=True, exist_ok=True)

def clear_old_dataset():
    if OUT_ROOT.exists():
        print(f"Removing old dataset folder: {OUT_ROOT.resolve()}")
        shutil.rmtree(OUT_ROOT)

def copy_images(files, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, src in enumerate(files, start=1):
        dst = out_dir / f"{src.parent.name}_{src.stem}{src.suffix.lower()}"
        if dst.exists():
            dst = out_dir / f"{src.parent.name}_{src.stem}_{i}{src.suffix.lower()}"
        shutil.copy2(src, dst)

def main():
    random.seed(42)

    total_train = 0
    total_val = 0
    total_test = 0
    total_all = 0

    # Optional: remove old dataset before re-creating
    clear_old_dataset()
    ensure_dirs()

    for cls, folder_name in CLASS_FOLDERS.items():
        src_folder = SOURCE_ROOT / folder_name

        if not src_folder.exists():
            raise FileNotFoundError(f"Missing source folder: {src_folder.resolve()}")

        imgs = list_images(src_folder)

        if len(imgs) == 0:
            raise RuntimeError(f"No images found in: {src_folder.resolve()}")

        random.shuffle(imgs)

        n = len(imgs)
        n_train = int(n * TRAIN)
        n_val = int(n * VAL)
        n_test = n - n_train - n_val

        train_files = imgs[:n_train]
        val_files = imgs[n_train:n_train + n_val]
        test_files = imgs[n_train + n_val:]

        copy_images(train_files, OUT_ROOT / "train" / cls)
        copy_images(val_files, OUT_ROOT / "val" / cls)
        copy_images(test_files, OUT_ROOT / "test" / cls)

        total_train += len(train_files)
        total_val += len(val_files)
        total_test += len(test_files)
        total_all += n

        print(
            f"Class {cls} ({folder_name}) -> "
            f"total={n}, train={len(train_files)}, val={len(val_files)}, test={len(test_files)}"
        )

    print("\n✅ Dataset split completed successfully")
    print(f"Output folder: {OUT_ROOT.resolve()}")
    print(f"Total images: {total_all}")
    print(f"Train: {total_train}")
    print(f"Validation: {total_val}")
    print(f"Test: {total_test}")
    print("\nNow run: python train_acne04.py")

if __name__ == "__main__":
    main()