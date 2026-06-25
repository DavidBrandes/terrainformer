# Terrainformer

[Intro]
[GIF]

## Usage
The application can be run with `make run` and configured via `config.toml`. Use the following controls to interact with the application:

| Input            | Action                          |
| ---------------- | ------------------------------- |
| Left/Right Mouse | Modify terrain                  |
| S                | Toggle shift tool               |
| C                | Toggle scene cropping to window |
| F                | Toggle fullscreen mode          |
| Q                | Quit the application            |

## Requirements
- CMake 3.24 or newer
- A C++20-compatible compiler
- CUDA Toolkit
- OpenGL 3.3-compatible drivers

## CUDA Kernel Optimization
### Profiling Hardware
The CUDA kernels in this application were profiled and optimized on an NVIDIA RTX 2000 Ada Generation Laptop GPU. The table below lists key hardware properties.

| Specification                   | Value       |
| ------------------------------- | ----------- |
| Compute capability              | 8.9         |
| Arithmetic throughput           | 12 TFLOPS   |
| Peak memory bandwidth           | 238.4 GiB/s |
| Streaming Multiprocessors (SMs) | 24          |
| Cores per SM                    | 128         |
| Warp size                       | 32          |
| Max blocks per SM               | 24          |
| Max threads per SM              | 1536        |
| Max threads per block           | 1024        |
| L1/shared memory per SM         | 100 KiB     |
| Max shared memory per block     | 48 KiB      |
| Available registers per SM      | 65536       |
| Available registers per block   | 65536       |
| L2 cache                        | 32 MiB      |
| DRAM                            | 7.6 GiB     |
| Constant memory                 | 64 KiB      |

### Smoothstep Kernel

#### Overview

To warm up, we start with the relatively simple [smoothstep](src/compute/kernels/smoothstep.cu) kernel. This element wise kernel is used by the application on a circular region of radius $r$ around the user's click position $c$ on the height grid. It modifies a grid's vertex $v$ with height $h_v$ and distance $d_v=\Vert{v - c}\Vert_2$ to the click position $c$ by an amount $m_v$, which we compute as
$$
m_v = 3\alpha f_v^2 - 2\alpha f_v^3 ,\quad\text{where }
f_v = \begin{cases} 1 - \frac{d_v}{r}, & d_v < r \\ 0, & d_v \geq r\end{cases}\text{ and }\alpha \in \mathbb{R}.
$$
The value $\alpha$ is a scaling factor which controls by how much and in which direction the height should be modified. 

Each click  iteratively updates the height grid as $
h_v^i=h_v^{i-1}+m_v^i$, where $h_v^i$ and $m_v^i$ denote the height and modification of vertex $v$ after the $i$-th click, respectively. Notice how vertices outside the radius $r$ are unaffected, as $f_v=0$ implies $m_v=0$.

To benchmark this kernel, we simulate a click centered on a grid of $8000\times4000$ vertices with a radius of half the grid's height. We start profiling with a naive kernel implementation that computes one vertex per thread and uses blocks of size $16\times16$. From there, we work toward a more performant variation.

#### Roofline Analysis

Already, we can see that in this naive implementation, most threads don't do any work. Their corresponding vertices lie outside the click's effect radius and hence will evaluate to a modification $m_v$ of $0$. In particular, we can estimate that only a fraction of $\pi r^2/8r^2\approx 0.39$ of all threads will be doing useful work.

We can also estimate the kernel's theoretical performance. For a vertex whose height is modified, computing $f_v, m_v$ and the height update requires 16 floating point operations. Compared to one load and one store totalling 8 bytes, this gives a computational intensity of $2\frac{\text{FLOP}}{\text{B}}$.

Even though the achievable computational throughput is significantly lower than the reported 12 TFLOPS (NCU reports a ceiling of ~5.7 TFLOPS, possibly due to thermal throttling), this kernel is clearly memory bound. Taking the measured 5.7 TFLOPS as reference, even a perfect kernel would use only ~9% of the GPU's compute capacity.

#### Avoiding powf

A simple, but very impactful optimization is to replace `powf` with explicit multiplication for computing squares and cubes. Unlike the explicit products, `powf` does not optimize integer exponents and generates significantly more instructions. This single change reduces issued instructions from 218,676,401 to 40,655,248.

#### Restricting the kernel

The next optimization is to restrict the kernel launch to the bounding square of side length $2r$ enclosing the click circle. This avoids computation on vertices that are guaranteed to be unaffected and reduces the kernel's runtime from 785 µs to 571 µs, increasing the fraction of threads doing useful work from 0.39 to $\pi r^2/4r^2 \approx 0.79$.

Since there are no cross dependencies between the vertices in our grid and we are mostly memory bound, we cannot do much except optimizing access to memory. Switching the existing block configuration of $16\times16$ threads to $32×8$, we observe a slight improvement in performance. Swapping this configuration to $8\times32$ and adding in vector loads along the rows, we are down to a runtime of 541 µs. 

This configuration gives us a square access of $32\times32$ elements per block, which, for specific brush radii, will give us the minimal amount of blocks required to cover the entire click's circle. We observe an increased throughput of between 29% and 70% in L1, L2 and DRAM when compared to an unvectorized implementation. We now utilize memory bandwidth at 92% and get more active warps on average. 

Using `float4` loads and stores, we effectively coarsen the kernel. This gives each thread more work and keeps it busy while others wait on their data to arrive, all while issuing fewer memory instructions than an unvectorized implementation. This modification brings us very close to what can be achieved with the GPU. Using the stated memory bandwidth of 256 GB/s, an optimal kernel would take 500 µs for the grid of $4000\times4000$ elements (considering both loads and stores).

#### Cache usage

When using this kernel during an actual application run, we can expect it to be launched multiple times in a row. On our GPU with a L2 cache size of 32 MiB, we can fit a total of 8,388,608 floating point values. In an optimal scenario, where the cache is not disturbed by anything else and the kernel is launched consecutively with a constrained block of $4000\times4000$ elements, around 52% of all data could theoretically be served hot from cache instead of DRAM.

To verify this behaviour, we benchmarked 100 identical runs of this kernel with and without flushing the cache before each run. We could not observe any meaningful difference in runtime. This suggests the kernel's access pattern evicts its own cache lines before they can be reused by a subsequent run. Indeed, if we run the kernel with a smaller brush radius that fits all data into L2, we see the overall runtime decreased by 28%.

#### Further ideas

In theory we could also eliminate the square root in the computation of the distance of a grid vertex to the click center. However leaving it out would give us only very small modifications at the circle boundaries. Even if that were acceptable, it would not give us much with our memory bound kernel. Fewer compute steps would simply leave the processors run idle more often.

### Marching Squares Kernel

#### Overview

We next take a look at the [marching squares](src/compute/kernels/marching_squares.cu) kernel. This kernel uses the [marching squares algorithm](https://en.wikipedia.org/wiki/Marching_squares) to compute the contour lines of the underlying height grid. For the specified threshold values, this algorithm approximates respective contours with linear segments at a resolution of the grid underlying dimensions.

Unlike a convolution kernel, this algorithm has a dynamic output size. For every four adjacent grid vertices (a $2\times2$ subgrid) and a single threshold, it may produce zero, one or two contour segments. This variance naturally increases with the amount of thresholds one wants to compute the contours at.

A simple solution to deal with this dynamic output would be to have each $2\times2$ subgrid always produce its maximal amount of output segments, setting the unused ones to `NaN` or some values outside the displayed area. However, that would take away a lot of the challenges for optimizing the kernel and what we are actually interested in. Plus, a kernel that yields only as many output segments as actually required gives a nice general solution and takes away work from the shader that renders them.

As before, we start with a naive implementation and will work our way towards a more performant version. We choose a block size of $16\times16$, use no shared memory and write the segments to an output buffer indexed via a grid scoped atomic counter. For the underlying height grid we again choose a size of $8000\times4000$. We also compute the contours for each threshold level sequentially one after another.

#### Analysis

Let us take a look at the computational intensity of this kernel. Since the amount of computational steps, bytes loaded and stored differs with the amount of output segments, we present the different cases in the below table.

|                  | 0 Segments | 1 Segment | 2 Segments |
| ---------------- | ---------- | --------- | ---------- |
| **Bytes Loaded** | 16         | 20        | 20         |
| **Bytes Stored** | 0          | 20        | 36         |
| **FLOPs**        | 0          | 8         | 38         |
| **OP/B**         | 0          | 0.2       | 0.68       |

We can already notice that in this naive implementation, the kernel is heavily memory bound, even more so than the previous smoothstep kernel. Further we see that the computational intensity differs by quite a bit across the various output conditions.

To get a feeling about the frequency, with which we can expect each of these 3 cases to arise, we launch the kernel on very regular sinusoidal grid and two variations of Perlin noise (one producing very turbulent and the other more gentle terrain). We also let the contour thresholds vary between the range of possible values. When counting the occurrence of each case, we notice a very skewed distribution which we show in the below table.

|                            | 0 Segments | 1 Segment | 2 Segments |
| -------------------------- | ---------- | --------- | ---------- |
| **Sinusoidal**             | ~99.9%     | ~0.1%     | ~0%        |
| **Gentle Perlin Noise**    | ~99.8%     | ~0.2%     | ~0%        |
| **Turbulent Perlin Noise** | ~99.1%     | ~0.89%    | ~0.01%     |

The output is quite revealing. For most $2\times2$ subgrids, we won't be doing any computation at all. Only rarely, a subgrid will actually produce an output segment and two are rarer still. As a grid initialized with gentle Perlin noise appears to be the more interesting and realistic case, we will use it to continue with our profiling.

#### Splitting the kernel into two
With this naive single-kernel implementation, we can expect warp divergence to be quite high. Using the above numbers we can expect one average only one thread in a grid of size 512 to do any floating point computations. That is, only one thread might prevent its whole block from returning early. Precious time that could be spend on computing the contours on different parts of the height grid instead.

Instead of computing the contours in one go, we can split our kernel into two parts. The first one will check each grid cell if a contour segment falls into it. If it does, the grid indices are written to a temporary buffer (of `int2`'s for convenience). As before the respective indices are being written by atomically increasing a global counter. The second kernel then loads each such pair of indices, computes the respective contour segment and, reusing the same index, writes them to the final contour buffer.

Launching the second kernel with the same amount of threads as the first, leaving the counter on the GPU, we already see a big improvement of several µs. However that still leaves the kernel with a huge amount of wasted threads. Instead we can copy the counter back to the CPU and only launch the kernel with the necessary amount of threads. Even though we now have a costly memory operation, the reduced kernel size is able to make up for up, giving us again a noticeable improvement in overall runtime.

While in theory we are still left with some warp divergence in the second kernel, the event of having two segments is so rare that we basically can ignore it. Further dividing the kernel introduces additional overhead that outweighs any performance gains from reducing negligible warp divergence.

#### Shared memory

As a benefit of having two kernels, we can now better profile individual sections. An obvious optimization seems to be the use shared memory instead of loading all the data individually from DRAM in each thread. With computation happening in $2\times2$ subgrids, most data is actually reused by other threads. With square blocks of size $n$, the amount of repeatedly loaded halo cells is with $4n-1$ fairly low compared to $n^2$, the amount of inner cells that are loaded only once. Using shared memory, the number of bytes loaded by such a block approaches one quarter of the original amount as $n\to\infty$.

Trying it in practice, we however observe a performance far worse than that of a naive implementation loading all data separately. The use of shared memory now adds additional instructions (due to the branching logic when loading data) and more importantly, barriers to our kernel. We now see way more warps stalling and doing nothing waiting for their block's data to arrive. This is especially costly with the large number of threads that later end up doing no useful work anyways.

#### TOOD

// store grid heights as float4
// 2D -> 1D; thread, warp, block scan -> atomic write (both on full and slim kernel)
// TODO try __restrict__, __ldg()

