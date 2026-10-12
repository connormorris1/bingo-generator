import cv2
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random

def read_imgs(path: str | Path, target_size: int) -> list[np.ndarray]:
    imgs_path = Path(path)
    imgs = []
    for img_path in imgs_path.iterdir():
        img = np.array(Image.open(img_path).convert("RGB"))
        if img.shape[0] < img.shape[1]:
            img = img[:,-img.shape[0]:,:]
        if img.shape[0] > target_size:
            img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
        imgs.append(img)
    return imgs

def create_board(imgs: list[np.ndarray], rows: int=4, cols: int=5, random_seed: int=None) -> np.ndarray:
    if len(imgs) != rows * cols:
        raise ValueError(f"Number of images ({len(imgs)}) does not match the board size ({rows}x{cols}={rows*cols}).")
    if random_seed is not None:
        random.seed(random_seed)
    random.shuffle(imgs)
    fig, ax = plt.subplots(rows,cols,figsize=(10,8))
    for i in range(rows):
        for j in range(cols):
            img = imgs[i*cols+j]
            ax[i,j].imshow(img)
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])
    fig.set_layout_engine(layout='constrained', w_pad=0, h_pad=0, hspace=0, wspace=0)
    fig.canvas.draw()
    rgba_buffer = fig.canvas.buffer_rgba()
    img_array = np.asarray(rgba_buffer)  # Shape will be (Height, Width, 4)
    img_array_rgb = img_array[:, :, :3]
    plt.close(fig)
    return img_array_rgb.copy()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create a board of images.")
    parser.add_argument("path", type=str, help="Path to the directory containing images.")
    parser.add_argument("--rows", type=int, default=4, help="Number of rows in the board (default: 4).")
    parser.add_argument("--cols", type=int, default=5, help="Number of columns in the board (default: 5).")
    parser.add_argument("--num_boards", type=int, default=1, help="Number of boards to create (default: 1).")
    parser.add_argument("--output_dir", type=str, default="boards/", help="Output path for the board image (default: 'boards/').")
    parser.add_argument("--target_size", type=int, default=256, help="Target size for resizing images (default: 90).")
    parser.add_argument("--random_seed", type=int, default=None, help="Random seed for shuffling images (default: None).")
    args = parser.parse_args()
    
    path = Path(args.path)
    target_size = args.target_size
    imgs = read_imgs(path, target_size)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.num_boards):
        board = create_board(imgs, rows=args.rows, cols=args.cols, random_seed=args.random_seed)
        output_path = output_dir / f"board_{i+1}.png"
        Image.fromarray(board).save(output_path)
        print(f"Saved board {i+1} to {output_path}")
