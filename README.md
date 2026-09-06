# Terrainformer

Terrainformer is an interactive terrain editor written in C++20, CUDA, and OpenGL. It lets the user raise and lower the terrain and watch its contour lines update in response. Both the height modification and contour computation are implemented as CUDA kernels.

The kernels sit at the heart of the application and are optimized specifically for how they are used here. The accompanying [CUDA kernel optimization journal](PROFILING.md) documents their path from naive implementations to the current ones through small, incremental profiling and optimization steps. It contains a few surprising results: small changes that made a big difference, seemingly sensible optimizations that made things slower, and even a case where giving a kernel more work made it run faster.

[DEMO]

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
