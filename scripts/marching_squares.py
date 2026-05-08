import numpy as np
from pathlib import Path
from noise import pnoise2

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt

def plot(grid, segments = [], title = "image"):
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
    plt.savefig(Path(__file__).parent / f"{title}.png", dpi=150, bbox_inches="tight")


def get_cell_type(top_left, top_right, bottom_right, bottom_left, threshold):
    cell_type = 0
    if top_left > threshold:
        cell_type |= 8
    if top_right > threshold:
        cell_type |= 4
    if bottom_right > threshold:
        cell_type |= 2
    if bottom_left > threshold:
        cell_type |= 1

    return cell_type



def marching_squares(grid, threshold):
    height, width = grid.shape
    segments = []

    for i in range(height - 1):
        for j in range(width - 1):
            top_left = grid[i, j]
            top_right = grid[i, j + 1]
            bottom_left = grid[i + 1, j]
            bottom_right = grid[i + 1, j + 1]

            cell_type = get_cell_type(top_left, top_right, bottom_right, bottom_left, threshold)
            cell_segments = get_segments(cell_type, i, j, top_left, top_right, bottom_left, bottom_right, threshold)

            segments.extend(cell_segments)

    return segments

def marching_squares_segment_types(grid, threshold):
    cell_type = (grid[:-1, :-1] > threshold) * 8 + (grid[:-1, 1:] > threshold) * 4 + (grid[1:, 1:] > threshold) * 2 + (grid[1:, :-1] > threshold) * 1

    types = {}
    types["0"] = int(np.sum(np.isin(cell_type, [0, 15])))
    types["1"] = int(np.sum(np.isin(cell_type, [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14])))
    types["2"] = int(np.sum(np.isin(cell_type, [5, 10])))

    return types


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

def make_perlin_grid(width, heigth, scale):
    grid = np.zeros((heigth, width))

    for y in range(heigth):
        for x in range(width):
            grid[y, x] = pnoise2(
                x / scale,
                y / scale,
                octaves=6,
                persistence=0.5,
                lacunarity=2.0,
                repeatx=1024,
                repeaty=1024,
                base=0
            )
    
    return grid

def make_sinusoidal_grid(width, height, frequency, amplitude):
    factor = min(height, width) - 1

    x = np.arange(width) / factor
    y = np.arange(height) / factor

    X, Y = np.meshgrid(x, y)
    grid = amplitude * np.sin(X * np.pi * frequency) * np.cos(Y * np.pi * frequency)

    return grid

if __name__ == "__main__":
    width, height = 8000, 4000
    frequency = 4
    amplitude = 1
    n_thresholds = 21
    n_vertices = (width - 1) * (height - 1)

    grid = make_perlin_grid(width, height, 80)
    # grid = make_grid(width, height, frequency, amplitude)
    thresholds = np.linspace(-amplitude, amplitude, n_thresholds + 2)

    summed_types = {"0": 0, "1": 0, "2": 0}

    for threshold in thresholds[1:-1]:
        types = marching_squares_segment_types(grid, threshold)

        summed_types["0"] += types["0"]
        summed_types["1"] += types["1"]
        summed_types["2"] += types["2"]

    summed_types["0"] /= (n_thresholds * n_vertices)
    summed_types["1"] /= (n_thresholds * n_vertices)
    summed_types["2"] /= (n_thresholds * n_vertices)

    print(summed_types)

# {'0': 0.9986729294733568, '1': 0.0013270228897332324, '2': 4.7636909971633544e-08} sinus
# {'0': 0.9910364350742166, '1': 0.008948316648632151, '2': 1.5248277151232576e-05} perlin


