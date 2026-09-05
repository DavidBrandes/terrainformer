# Terrainformer

[TODO: Intro]

[TODO: GIF]

## CUDA Optimization Journal

[TODO: Optimization intro]

[Read the CUDA kernel optimization journal](PROFILING.md).

## Build and Run

### Requirements

- Linux with a CUDA-capable NVIDIA GPU
- CMake 3.24 or newer
- A C++20-compatible compiler
- CUDA Toolkit
- OpenGL 3.3-compatible drivers

```sh
make run
```

The application can be configured via [`config.toml`](config.toml).

## Controls

| Input            | Action                          |
| ---------------- | ------------------------------- |
| Left/Right Mouse | Raise/lower terrain             |
| Mouse Wheel      | Change brush radius             |
| S                | Toggle the terrain tool         |
| C                | Toggle scene cropping           |
| F                | Toggle fullscreen mode          |
| Q                | Quit                            |

## License

Licensed under the [MIT License](LICENSE).
