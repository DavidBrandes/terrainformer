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

This algorithm may be viewed as a combination of a convolution and an (unstable) filter kernel. Like a convolution kernel, we stride over the whole input grid with $2\times2$ subgrids. However, unlike a convolution, each subgrid produces a variably sized output For every four adjacent grid vertices (a $2\times2$ subgrid) and a single threshold, it may produce zero, one or two contour segments.

A simple solution to deal with this dynamic output would be to have each $2\times2$ subgrid always produce its maximal amount of output segments, setting the unused ones to `NaN` or some values outside the displayed area. However, that would take away a lot of the challenges for optimizing the kernel and what we are actually interested in. Plus, a kernel that yields only as many output segments as actually required gives a nice general solution and takes away work from the shader that renders them.

As before, we start with a naive implementation and will work our way towards a more performant version. We choose a block size of $16\times16$, use no shared memory and write the segments to an output buffer indexed via a grid scoped atomic counter. For the underlying height grid we again choose a size of $8000\times4000$. For now, we compute the contours only for a single threshold sitting right at the middle of the height grid's range. Later on, we will also look at the case of multiple thresholds.

#### Analysis

Let us take a look at the computational intensity of this kernel. Since the amount of computational steps, bytes loaded and stored differs with the amount of output segments, we present the different cases in the below table.

|                  | 0 Segments | 1 Segment | 2 Segments |
| ---------------- | ---------- | --------- | ---------- |
| **Bytes Loaded** | 16         | 20        | 20         |
| **Bytes Stored** | 0          | 20        | 36         |
| **FLOPs**        | 0          | 8         | 38         |
| **OP/B**         | 0          | 0.2       | 0.68       |

We can already notice that in this naive implementation, the kernel is heavily memory bound, even more so than the previous *smoothstep kernel*. Further we see that the computational intensity differs by quite a bit across the various output conditions.

To get a feeling for the frequency with which we can expect each of these three cases to arise, we launch the kernel across several grid configurations. In particular, we look into three categories: regular sinusoidal grids, and two groups of Perlin noise, one producing more gentle and the other more turbulent terrain. For each category we vary parameters and contour threshold across their respective ranges. The distribution of the three output cases for each category is displayed in the below table. The output is quite revealing. We observe a very skewed distribution as for most $2\times2$ subgrids, we won't be doing any computation at all. Only rarely, a subgrid will actually produce an output segment and two are rarer still.

|                            | 0 Segments | 1 Segment | 2 Segments |
| -------------------------- | ---------- | --------- | ---------- |
| **Sinusoidal**             | ~99.9%     | ~0.1%     | ~0%        |
| **Gentle Perlin Noise**    | ~99.8%     | ~0.2%     | ~0%        |
| **Turbulent Perlin Noise** | ~99.1%     | ~0.89%    | ~0.01%     |

As a grid initialized with gentle Perlin noise appears to be the more interesting and realistic case, we will use it to continue with our profiling (parameters: octaves 4, frequency 6, persistence 0.5). Since this grid produces most of its segments close to the midpoint of its height range, we choose this midpoint as our contour threshold. At this threshold, our selected initialization produces only 0 and 1, but no 2, output segments per $2\times2$ subgrid. The actual counts are shown below.

|                | 0 Segments | 1 Segment | 2 Segments |
| -------------- | ---------- | --------- | ---------- |
| **Count**      | 31,853,485 | 134,516   | 0          |
| **Occurrence** | ~99.6%     | ~0.4%     | 0%         |

#### Using vector stores for the output
In its basic implementation, the performance of our kernel is quite poor. We measure a runtime of 2.76 ms on our $8000\times4000$ profiling grid. However there is an easy win waiting for us. Each contour segment consists of two $(x, y)$ start and end points, which maps naturally onto a single `float4` vector.

We already saw in the smoothstep kernel how vector stores and loads improved performance. In this kernel the gain is even more pronounced. Switching to `float4` stores, we observe a speedup by a factor of 1.33. The runtime is now at 2.07 ms and the number of executed instructions decreased by 37% (although we suspect that some of these gains can be attributed to the fact that we now construct the output vectors in place, while before we constructed the output data upfront and only later chose what was needed).

#### Splitting the kernel into two
With the naive single-kernel implementation, we can expect warp divergence to be quite high. From the distribution measured above, on average only around one thread in every 500 will perform any floating-point computation. A single active thread is enough to prevent its entire warp from retiring early. Precious execution time that could otherwise be spent computing contours on a different region of the height grid.

Instead of computing the contours in a single pass, we can split our kernel into two. The first kernel checks each grid cell for the presence of a contour segment. If one is found, the cell's grid indices are written to a temporary buffer (using `int2`'s for convenience). As in the basic implementation, the output slot is determined by atomically increasing a global counter. The second kernel then reads each such index pair, computes the corresponding contour segment, and writes it, reusing the same index, to the final contour buffer.

To profile this modified setup, we need to switch to CUDA events as opposed to Nsight Compute, the profiling tool that we've used so far. Because Nsight Compute introduces profiling overhead by its nature and may rerun kernels several times to gather all information, the reported runtimes may differ from those measured with CUDA events and are typically higher. The conclusions we make about the relative kernel runtimes, however, hold regardless.

CUDA events allow us to capture timings across multiple kernel launches, including any potential memory transfers in between. In our setup, we run each variant 100 times after 5 warmup iterations and report the average runtime. Using this method, we observe an average runtime of 1429 µs for the current version of our kernel before applying any division.

In a first attempt, we launch the second kernel with the same amount of threads as there are potential contour lines, leaving the counter resident on the GPU. This yields a degraded runtime of 1551 µs (1137 µs for first and 414 µs in the second kernel). The second kernel wastes the vast majority of its threads doing nothing.

We can do better by copying the counter back to the CPU and launching the second kernel with exactly as many threads as there are segments to compute. Despite the added cost of the device-to-host transfer, the much smaller dispatch more than compensates, improving the overall runtime down to 1194 µs (using 17 µs and 40 µs for the memory transfer and second kernel respectively).

Splitting the kernel into two parts, we improved our algorithm by approximately 16%. In theory, we are still left with warp divergence in the second kernel. But the two-segment case is so rare, that we can essentially neglect it. Subdividing the kernel further would introduce additional overhead that outweighs any gains from eliminating this remaining divergence.

#### Storing the height grid in shared memory
As a further benefit of having two kernels, we can now profile each stage. An obvious next optimization is shared memory. With computation happening in $2\times2$ subgrids, most data is actually reused by other threads. With square blocks of size $n$, the amount of repeatedly loaded halo cells is $4n-1$. A quite low amount when compared to $n^2$, the amount of inner cells that are loaded only once. Using shared memory, the number of bytes loaded by such a block approaches one quarter of the original amount as $n\to\infty$.

Trying it in practice, we however observe a performance far worse than that of a naive implementation loading all data separately. The use of shared memory now adds additional instructions (due to the branching logic when loading data) and more importantly, barriers to our kernel. We now see way more warps stalling and doing nothing waiting for their block's data to arrive.

#### Packing the grid values for reuse
We saw that even though there are repeated loads, the first kernel still exhibits a favorable memory access pattern. The second kernel, in its current form, unfortunately doesn't. Memory accesses can be scattered around the height grid and two neighboring threads need not access neighboring data. As a result, the L2 cache hit rate is only around 57.5%. This is a relatively low value considering that each thread accesses two pairs of consecutive data elements by default.

Since data reuse is limited, we apply the same idea as before and store the corresponding $2\times2$ subgrid values as packed `float4` vectors in the first kernel, alongside the grid indices. The second kernel can then load these packed values instead of fetching the data from the height grid directly. While this introduces an additional store instruction in the first kernel, it allows the second kernel to access its input in a fully coalesced manner.

Although this optimization reduces the runtime of the second kernel by approximately 39%, it also increases the runtime of the first kernel slightly by about 1% due to the additional stores. Overall, the combined runtime appears to be marginally lower than that of the original implementation. However, we were unable to reproduce this improvement consistently across repeated measurements. Given the additional implementation complexity and the lack of a reproducible speedup, we decided not to pursue this optimization further.

#### Reducing pressure on the atomic counter
We need to find another way to optimize our kernels further. Looking at the warp stall statistics of the first, we observe a very high cycle count for *Stall Long Scoreboard*. This metric indicates that our kernel's warps spend a lot of their lifetime waiting on data from DRAM to arrive. This might be due to the kernel's general memory requirements but inspecting the code we actually see another culprit. Every thread increments the global atomic counter for the number of output segments, regardless of whether its increment is zero.

All of these operations put a lot of pressure on the counter. After modifying the code to only increment the counter when a segment is actually produced (an optimization nvcc cannot perform itself, as it may not elide atomic operations), we find the above stall metric almost halved, now averaging only around 6 cycles. Consequently, the amount of *Eligible Warps Per Scheduler* is now increased by 73% to 1.53. Overall, reducing the contention on the atomic variable decreased the runtime of the first kernel from 1.81 ms to 1.06 ms.

We could try to go further, aggregate the increments, and only write to the counter once per warp/block. However, this does not give us much at this point. As we saw in the analysis, the case of having a grid vertex produce a contour segment is very rare. So rare that any synchronization actually degrades performance. We illustrate the runtime of the kernel variants in the below table.

|             | Write from each thread | Write if > 0 | Write once per warp | Write once per block |
| ----------- | ---------------------- | ------------ | ------------------- | -------------------- |
| **Runtime** | 1.81 ms                | 1.06 ms      | 1.49 ms             | 1.25 ms              |

#### Computing contours at multiple thresholds
So far we computed our height grid's contours only at a single threshold. For our specific use case we, however, want to display contours at various thresholds. Our chosen threshold sits at the midpoint of the grid's height range. The further we move a threshold towards the range's bounds, the fewer output segments it will produce. The below table illustrates the amount of output segments for evenly spaced thresholds across the grid's height range. To profile the marching squares kernel in the multi-threshold setup, we will use 100 values, evenly spaced across the same range. This produces a total of 4,753,797 output segments.

|              | $-\frac{3}{5}$ | $-\frac{2}{5}$ | $-\frac{1}{5}$ | $0$     | $\frac{1}{5}$ | $\frac{2}{5}$ | $\frac{3}{5}$ |
| ------------ | -------------- | -------------- | -------------- | ------- | ------------- | ------------- | ------------- |
| **Segments** | 662            | 7,540          | 69,385         | 134,516 | 76,476        | 10,044        | 86            |

We further make two assumptions: we will only know the exact number of thresholds at runtime (meaning we cannot place them into constant memory) and do not care about the ordering in which the output segments are produced. The latter opens the way for optimizations as it loosens restrictions on how data should be stored. However, it also takes away the option to differentiate between contours at different thresholds, e.g. if we wanted to color each one differently.

#### Enabling parallelism across multiple thresholds
What currently blocks us from computing contours at multiple thresholds in parallel is that we do not know which grid indices correspond to which threshold. From the first kernel, we currently save only the indices at which the second kernel then computes the contour segments. If we were to compute multiple thresholds in parallel, the second kernel would not know which threshold to use. Without any modifications, this leaves us naively computing the segments sequentially, one threshold after another.

However, that would leave several optimizations underutilized. To enable them, we can create a second temporary output buffer, to which the first kernel stores the corresponding thresholds. Each element of this buffer corresponds to an index pair in the temporary vertex buffer. The second kernel then loads both buffers and is consequently able to identify a grid index pair together with its threshold.

#### Three approaches to execute in parallel
With this tweak, we are now able to launch the first kernel in batch for all thresholds simultaneously. Each launch is assigned to a dedicated CUDA stream, in which it can execute independently of the others. The second kernel then only has to run once, after all previous ones have finished.

We are still left with quite some kernel launches. Even though we run them in parallel, the first kernel is launched 100 times. CUDA Graphs are a great way to speed up such a situation. Instead of launching each kernel individually from the host, we can capture the sequence of launches into a graph once and then replay this graph as a single unit for every subsequent call. This removes most of the CPU-side launch overhead that would otherwise accumulate.

To avoid these multiple launches of the first kernel altogether, we can also vectorize them. If we make the thresholds a third dimension after the grid's height and width, we can launch a single kernel on a 3-dimensional grid and compute all thresholds together. This approach brings us back to two kernel launches, but now computing contours at multiple thresholds. The first, 3-dimensional kernel populates the temporary index and threshold buffers. The second kernel then computes the segments from these.

As a consequence of this vectorization, we can no longer pass the thresholds directly as the first kernel's parameters. Since we only know the threshold count at runtime, this leaves us no option other than passing a pointer to them instead. Each thread then has to load its threshold dynamically at runtime.

#### Comparing the four approaches
The performance of these four approaches is reported in the table below. In the first row, we show the total runtime for each, spanning both the (potentially multiple) first kernel launches and the second kernel launch. As before, these times were measured using CUDA events and calculated as the average of 100 runs after 5 warmup iterations. In the second and third rows, we report the runtime of the first and second kernels respectively. These were measured using Nsight Compute, with a single threshold placed at the height grid's midpoint.

|                   | Naive    | Batched  | Batched & CUDA Graphs | Vectorized |
| ----------------- | -------- | -------- | --------------------- | ---------- |
| **Combined**      | 86.86 ms | 95.61 ms | 87.26 ms              | 92.44 ms   |
| **First Kernel**  | 1.04 ms  | 1.05 ms  | 1.05 ms               | 1.13 ms    |
| **Second Kernel** | 0.05 ms  | 0.05 ms  | 0.05 ms               | 0.05 ms    |

We observe an unexpected behavior. None of the modifications were actually able to improve the overall runtime. The processors are already busy computing contours at one threshold, so our effort to parallelize could not help much. Also, the batched version's additional DRAM storage and stream management overhead actually seem to be working against it. Using CUDA Graphs, we are able to make up for this somewhat, but still do not manage to outperform the naive implementation.

The vectorized approach performs rather poorly as well. Even though it only launches two kernels, its runtime is worse than the naive version's. In particular, the added dependency of indirectly loading the thresholds seems to hurt the first kernel's performance. We can also see that the second kernel plays only a minor role in the overall setup.

#### Coarsening the threads along the thresholds
Despite its initially weak performance, we decide to continue with the vectorized implementation. Its shape gives us a huge advantage: the ability to coarsen threads along the new threshold dimension. Instead of having one thread compute segments at one $2\times2$ subgrid and one threshold, we can instead let it do so for multiple thresholds. The subgrid stays fixed and doesn't need to be reloaded throughout these computations. This saves us costly time, as the thread now only stalls once waiting for data from DRAM, instead of stalling for each threshold as in the un-coarsened version.

We measured the runtimes of the vectorized kernel at various coarsening factors and show them in the table below. As before, we report them once for the individual kernels, measured with Nsight Compute, and once for their combination, measured with CUDA events. Starting already at a coarsening factor of two, we can see the runtime rapidly declining, outperforming the naive version considerably. Please note that before, we profiled the individual kernels at only one threshold, while now we do so for all 100, hence the difference in runtime.

| Coarse Factor     | 1         | 2        | 4        | 8        | 16       | 32       | 64       | 128      |
| ----------------- | --------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| **Combined**      | 93.45 ms  | 59.58 ms | 42.61 ms | 34.78 ms | 30.81 ms | 28.31 ms | 27.37 ms | 25.69 ms |
| **First Kernel**  | 137.02 ms | 84.28 ms | 61.28 ms | 53.98 ms | 50.56 ms | 48.98 ms | 47.84 ms | 47.29 ms |
| **Second Kernel** | 1.73 ms   | 1.36 ms  | 1.13 ms  | 1.08 ms  | 1.07 ms  | 1.06 ms  | 1.06 ms  | 1.05 ms  |

The reduced memory traffic really paid off. As a consequence, we observe the L1 and L2 cache hit rates increase by 28% and 95% respectively, when comparing the un-coarsened version to the version coarsened by a factor of 128. Interestingly, the second kernel also improved. This kernel loads the height grid's data from the grid indices that were stored in the temporary buffer by the first kernel. The closer together the data represented by these indices are, the more efficiently they can be loaded from DRAM. As a consequence of the thread blocks staying longer in one area of the height grid, we get less variance in the resulting index buffer. An increased memory throughput of approximately 64% across all layers confirms this optimized pattern.

// Prefetching the thresholds
// Block write, privatization
// Outlook, Intro, Code snippets