import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt

def plot(grid, segments):
    plt.figure(figsize=(10, 8))

    plt.imshow(grid, cmap="viridis", alpha=0.5)
    plt.colorbar(label="Height")

    for x1, y1, x2, y2 in segments:
        plt.plot([x1, x2], [y1, y2], "r-", linewidth=2)

    plt.title("Marching Squares")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "contour.png", dpi=150, bbox_inches="tight")


def marching_squares(grid, threshold):
    height, width = grid.shape
    segments = []

    for i in range(height - 1):
        for j in range(width - 1):
            top_left = grid[i, j]
            top_right = grid[i, j + 1]
            bottom_left = grid[i + 1, j]
            bottom_right = grid[i + 1, j + 1]

            cell_type = 0
            if top_left > threshold:
                cell_type |= 8
            if top_right > threshold:
                cell_type |= 4
            if bottom_right > threshold:
                cell_type |= 2
            if bottom_left > threshold:
                cell_type |= 1

            cell_segments = get_segments(cell_type, i, j, top_left, top_right, bottom_left, bottom_right, threshold)
            segments.extend(cell_segments)

    return segments


def get_segments(cell_type, i, j, tl, tr, bl, br, threshold):
    segments = []

    def lerp(v1, v2):
        if abs(v2 - v1) < 1e-10:
            return 0.5
        
        return (threshold - v1) / (v2 - v1)

    top = j + lerp(tl, tr)
    right = i + lerp(tr, br)
    bottom = j + lerp(bl, br)
    left = i + lerp(tl, bl)

    if cell_type == 0 or cell_type == 15:
        pass

    elif cell_type == 1 or cell_type == 14:
        segments.append((j, left, bottom, i + 1))

    elif cell_type == 2 or cell_type == 13:
        segments.append((bottom, i + 1, j + 1, right))

    elif cell_type == 3 or cell_type == 12:
        segments.append((j, left, j + 1, right))

    elif cell_type == 4 or cell_type == 11:
        segments.append((top, i, j + 1, right))

    elif cell_type == 5:
        segments.append((j, left, top, i))
        segments.append((bottom, i + 1, j + 1, right))

    elif cell_type == 6 or cell_type == 9:
        segments.append((top, i, bottom, i + 1))

    elif cell_type == 7 or cell_type == 8:
        segments.append((j, left, top, i))

    elif cell_type == 10:
        segments.append((top, i, j + 1, right))
        segments.append((j, left, bottom, i + 1))

    return segments

if __name__ == "__main__":
    height, width = 50, 80
    threshold = 0.5

    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    X, Y = np.meshgrid(x, y)
    grid = 0.5 + 0.5 * np.sin(X * np.pi) * np.cos(Y * np.pi)

    segments = marching_squares(grid, threshold)
    plot(grid, segments)


