## CUDA Kernel Optimization
The [Terrainformer](README.md) application utilizes two CUDA kernels: a rather simple [smoothstep kernel](#smoothstep-kernel) that allows us to modify the terrain and a more complex [marching squares kernel](#marching-squares-kernel) that computes the corresponding contour lines. Since both kernels are executed whenever the terrain or its contours are updated, their performance directly affects how the application responds.

We will profile and optimize both of them, starting from naive implementations and iteratively work towards more efficient solutions. The two kernels present rather different challenges. The smoothstep kernel performs regular, element-wise operations and will eventually be limited mainly by memory bandwidth. The marching squares kernel, on the other hand, produces sparse and variably sized output, requiring special handling.

In each step, we will examine the underlying reasons behind the observed performance differences and support our explanations with metrics collected using Nsight Compute. Along the way, we will see simple code changes produce substantial speedups, seemingly suitable optimizations make kernels slower and additional computation unexpectedly improve performance.

The process we present will not always show a linear progression. Especially for the marching squares kernel, some approaches will not provide the expected performance gains or will become redundant after a subsequent modification. We still include them in this journal to show the honest path we took. Beyond that, we find the insights they provide interesting regardless of their eventual performance benefit.

We also try to keep each optimization step as minimal and self-contained as possible. Some steps, however, require refactoring the code, which may by itself slightly modify the kernel's behavior. In such situations, a performance increase or decrease might not be explained entirely by the optimization alone, but could also be influenced by the surrounding refactoring. We try to minimize these effects throughout our journey and explicitly point them out whenever we are aware of them.

### Profiling Hardware
The profiling and optimization were performed on an NVIDIA RTX 2000 Ada Generation Laptop GPU. The table below lists key hardware properties.

| Specification                   | Value       |
| ------------------------------- | ----------- |
| Compute capability              | 8.9         |
| Arithmetic throughput           | 12 TFLOPS   |
| Peak memory bandwidth           | 256 GB/s    |
| Streaming Multiprocessors (SMs) | 24          |
| Cores per SM                    | 128         |
| Warp size                       | 32          |
| Max blocks per SM               | 24          |
| Max threads per SM              | 1536        |
| Max threads per block           | 1024        |
| L1 cache + shared memory per SM | 128 KiB (100 KiB max for shared memory) |
| Shared memory per block (default / max) | 48 KiB / 99 KiB |
| Available registers per SM      | 65536       |
| Available registers per block   | 65536       |
| L2 cache                        | 32 MiB      |
| DRAM                            | 7.6 GiB     |
| Constant memory                 | 64 KiB      |

### Smoothstep Kernel

#### Overview

To warm up, we start with the relatively simple [smoothstep](src/compute/kernels/smoothstep.cu) kernel. This element wise kernel is used by the application on a circular region of radius $r$ around the user's click position $c$ on the underlying height grid. It modifies a grid's vertex $v$ with height $h_v$ and distance $d_v=\Vert{v - c}\Vert_2$ to the click position $c$ by an amount $m_v$, which we compute as
$$
m_v = 3\alpha f_v^2 - 2\alpha f_v^3 ,\quad\text{where }
f_v = \begin{cases} 1 - \frac{d_v}{r}, & d_v < r \\ 0, & d_v \geq r\end{cases}\text{ and }\alpha \in \mathbb{R}.
$$
The value $\alpha$ is a scaling factor which controls how much, and in which direction, the height is modified. 

Each click iteratively updates the height grid as $
h_v^i=h_v^{i-1}+m_v^i$, where $h_v^i$ and $m_v^i$ denote the height and modification of vertex $v$ after the $i$-th click, respectively. Notice how vertices outside the radius $r$ are unaffected, as $f_v=0$ implies $m_v=0$.

To benchmark this kernel, we simulate a click centered on a grid of width $8000$ and height $4000$ vertices with a radius of half the grid's height. We start the profiling journey with a naive kernel implementation that computes one vertex per thread and uses blocks of size $16\times16$. From there on, we will work toward more performant variations.

```C++
__global__ void smoothstep(float* heights, Size grid_size, Range height_range, BrushDab brush_dab) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int index = row * grid_size.width + col;
    
    if (col < grid_size.width && row < grid_size.height) {
        float x_diff = (float)col - brush_dab.x;
        float y_diff = (float)row - brush_dab.y;

        float distance = sqrtf(powf(x_diff, 2) + powf(y_diff, 2));

        float height = heights[index];
        float factor = 1 - distance / brush_dab.radius;
        height += (3 * powf(factor, 2) - 2 * powf(factor, 3)) * brush_dab.intensity;
        // clamp the height into the grid's allowed value range
        height = fmaxf(height_range.min, fminf(height_range.max, height)); 

        if (distance < brush_dab.radius) {
            heights[index] = height;
        }
    }
}
```

#### Roofline analysis
Looking at the code, we can already see that in this naive implementation, most threads won't be doing any work at all. Since their corresponding vertices lie outside the click's effect radius, the modification $m_v$ will evaluate to $0$ and hence the condition `if (distance < brush_dab.radius)` will not be satisfied. In particular, we can compute that only approximately $\pi r^2/8r^2 \approx 0.39$ of all threads will be doing useful work.

We can also make a naive estimate of the kernel's theoretical performance at an algorithmic level. For a vertex whose height is modified, computing $f_v, m_v$ and the height update requires 16 mathematical operations if we count `powf`, `sqrtf` and clamping each as a single operation. Compared to one load and one store totalling 8 bytes, this gives an operational intensity of $2\frac{\text{OP}}{\text{B}}$. Under this deliberately simplified model, the kernel appears memory bound. Using the measured compute ceiling of 5.7 TFLOPS and reported bandwidth of 238.4 GiB/s, a perfect implementation in which each counted operation maps to one FLOP would use only ~9% of the GPU's compute capacity. This estimate does not model the actual cost of operations such as `powf` in the compiled kernel.

#### Replacing powf with explicit multiplication

A simple, but very impactful, first optimization is to replace `powf` with explicit multiplication wherever we currently are computing squares and cubes. Unlike the explicit products, `powf` does not optimize integer exponents and generates significantly more instructions. This small change single handedly reduces issued instructions from 293,805,820 down to 55,979,098.

```C++
float distance = sqrtf(x_diff * x_diff + y_diff * y_diff);
height += factor * factor * (3 - 2 * factor) * brush_dab.intensity;
```

#### Hoisting the radius check

Another straightforward modification, which brings issued instructions further down to 39,837,711, is to move the distance check `if (distance < brush_dab.radius)` further upwards. This modifies the kernel to only perform the heavy computations and memory load if its threads lie inside the height brush's circle. We are unsure why NVCC did not perform this optimization by itself right away, since such a tweak appears rather plausible to us. Using Nsight Compute, we measure the runtime of the kernel adapted this way at 763 µs.

```C++
float x_diff = (float)col - brush_dab.x;
float y_diff = (float)row - brush_dab.y;

float distance = sqrtf(x_diff * x_diff + y_diff * y_diff);

if (distance < brush_dab.radius) {
    float height = heights[index];

    // Continue..
}
```

#### Avoiding unnecessary square root computations
After our previous success replacing `powf` with a simpler instruction, we might be tempted to do something similar with the `sqrtf` occurring in the distance computation. Ultimately the distance's square root is only needed inside the branch when computing the factor. Threads outside the brush circle could equally compare their squared distance $d_v^2$ against the squared radius $r^2$. Only when this condition holds do we need to take its root. The squared radius is fixed throughout the kernel's lifetime and can be passed along with its parameters.

```C++
float distance_sq = x_diff * x_diff + y_diff * y_diff;
if (distance_sq < brush_dab.radius_sq) {
    float distance = sqrtf(distance_sq);
    float factor = 1 - distance / brush_dab.radius;

    // Continue..
}
```

Testing it out, we however observe only a minor performance increase. While issued instructions are down by 15%, the kernel's runtime improved only by 4%. On modern GPUs, taking the square root is rather cheap and pretty optimized with dedicated special function units. So unfortunately we could not get an easy win as before. Having the upcoming optimization from the subsequent chapter already in mind, this effect will be even less pronounced. Considering the only minor performance gain but bloated function signature, we decide not to pursue this optimization any further.

#### Restricting the kernel to the brush's bounding square

We already noticed how, due to the height brush's nature, most threads won't be doing any work. To keep these numbers lower, we can restrict the kernel launch to only the bounding square of side length $2r$, minimally enclosing the height brush's click circle. This avoids computation on vertices that are guaranteed to be unaffected and reduces the kernel's runtime further down to 560 µs. With this modification, we stripped away lots of unnecessary threads and increased the fraction of the ones doing useful work from 0.39 to $\pi r^2/4r^2 \approx 0.79$.

```C++
Region region = restricted_brush_dab_region(brush_dab, grid_size);

if (!region.empty()) {
    dim3 block_dim(16, 16);
    dim3 grid_dim(ceil_div(region.size.width, block_dim.x), ceil_div(region.size.height, block_dim.y));

    smoothstep<<<grid_dim, block_dim>>>(heights, grid_size, height_range, brush_dab, region.origin);
}
```

The brush circle size we previously picked is rather big and covers lots of the height grid's space. We expect the brush's usage in the application to be considerably finer and restricted to much smaller areas of the grid. Seeing the above performance gain, we can expect even more during actual use. An experimental kernel launch with half our previous radius shows, for example, a further decrease in runtime to 125 µs.

#### Vectorizing memory access

Since there are no cross dependencies between the vertices in our grid and we are mostly memory bound, there isn't much we can do other than optimize those memory accesses. We could try to coarsen our kernel along either dimension or alter the grid configuration, but neither of these gives us much benefit. We only found a very small runtime decrease coarsening the kernel along the height dimension with a factor of 2, or modifying the block size to $32\times8$.

However, there is another small trick waiting for us. If we are ok with adding the requirement of 16-byte alignment and a width divisible by 4 to our height grid, we are able to use `float4` vector loads instead of scalar loads. Instead of loading only one height grid vertex, each thread will load four consecutive vertices in this modification. In a sense, this adds coarsening along the width dimension to our kernel, with the benefit of needing only one load instruction. To accommodate for the new requirements, we might need to extend the bounding square a bit to its right or left. This could potentially add up to three "unnecessary" columns in either direction. However, especially for bigger brush circles, we deem this to be ok.

```C++
__global__ void smoothstep(float* heights, Size grid_size, Range height_range, BrushDab brush_dab, Vertex offset) {
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * 4 + offset.col;
    int row = blockIdx.y * blockDim.y + threadIdx.y + offset.row;

    if (col < grid_size.width && row < grid_size.height) {
        int index = row * grid_size.width + col;
        // We guarantee grid_size.width and offset.col to be multiples of 4, 
        // hence index being 16-byte aligned
        float4 values = *(float4*)&(heights[index]);

        values.x = apply_brush_dab(values.x, col, row, height_range, brush_dab);
        values.y = apply_brush_dab(values.y, col + 1, row, height_range, brush_dab);
        values.z = apply_brush_dab(values.z, col + 2, row, height_range, brush_dab);
        values.w = apply_brush_dab(values.w, col + 3, row, height_range, brush_dab);

        *(float4*)&(heights[index]) = values;
    }
}
```

We can play around a bit with different block configurations but this won't be doing much difference. If anything, we can notice an improvement switching it to $8\times32$.  Seeing how this brings us back to a square access of $32\times32$ elements per block, we decide to keep it. A square access should give us, in theory, the minimal number of blocks required to cover the entire click's circle.

Compared to the unvectorized implementation, we observe a decrease in issued instructions by 27%. Memory throughput is now higher across all layers. Even more important for our memory-bound kernel, memory is now 87% busier than before, and we use 91% of our available bandwidth. All of this is a consequence of having more memory throughput per instruction. The kernel's runtime is now at 524 µs.

#### Reducing unnecessary memory access
While the improvement in the memory-related metrics from the previous modification seems impressive, the actual speedup is rather disappointing. In the current implementation, we load and store grid heights from DRAM unconditionally, even though they might not be modified. Given our previous computation, these wasted memory operations would occur in ~20% of the kernel's threads. This shows the metrics above did not tell the full story on their own. A busy memory pipeline transferring pointless bytes is not what we want.

We can do better than this. Adding a few lines to the code, we can first check if any of a thread's four vertices lie inside the brush's circle. Only if this condition holds true, we then load the values from memory. Profiling this variation with Nsight Compute, we now find the Streaming Multiprocessors busy 30% more often than before. We get around 61% more eligible and around 30% more issued warps per scheduler, on average. This is the direct benefit of the boundary warps no longer needing access to memory. The kernel's overall runtime is now at 408 µs.

 ```C++
float distance_x = compute_distance(col, row, brush_dab);
float distance_y = compute_distance(col + 1, row, brush_dab);
float distance_z = compute_distance(col + 2, row, brush_dab);
float distance_w = compute_distance(col + 3, row, brush_dab);

bool active_x = is_active(distance_x, brush_dab);
bool active_y = is_active(distance_y, brush_dab);
bool active_z = is_active(distance_z, brush_dab);
bool active_w = is_active(distance_w, brush_dab);

if (active_x || active_y || active_z || active_w) {
    int index = row * grid_size.width + col;
    float4 values = *(float4*)&(heights[index]);

    if (active_x) {
        values.x = apply_brush_dab(values.x, distance_x, height_range, brush_dab);
    }
    if (active_y) {
        values.y = apply_brush_dab(values.y, distance_y, height_range, brush_dab);
    }
    if (active_z) {
        values.z = apply_brush_dab(values.z, distance_z, height_range, brush_dab);
    }
    if (active_w) {
        values.w = apply_brush_dab(values.w, distance_w, height_range, brush_dab);
    }

    *(float4*)&(heights[index]) = values;
}
 ```

#### Cache usage

When using this kernel during an actual application run, we can expect it to be launched multiple times in a row in short sequences. On our GPU with an L2 cache size of 32 MiB, we can fit a total of 8,388,608 floating-point values. In our setup we launch the kernel with a brush whose circle covers the entire grid's height. This gives us approximately $4000^2*\frac{\pi}{4}=4\pi*10^6\approx12,566,370$ floating point values which the kernel operates on. This means around 67% of all values could theoretically be served hot from cache instead of being loaded from DRAM. 

To verify this behavior, we benchmarked 100 identical runs of this kernel with and without flushing the cache in between the runs. With the existing setup, i.e. using a brush radius of half the grid's height, we observed only a very slight speedup of 1.06. This suggests the kernel's access pattern largely evicts its own cache lines before they can be reused by a subsequent run. Indeed, if we run the kernel with a smaller brush radius that comfortably fits all data into L2 ($r=1500$), we get a speedup of 1.43, meaning a reduction in runtime by about 30%.

#### Summary

With this memory-bound kernel, the current implementation approaches the memory-bandwidth limit for our benchmark. In the previous section, we estimated our kernel to modify approximately $4\pi*10^6$ grid elements. Taking the stated bandwidth of 256 GB/s and the fact that for each element we need 8 bytes of data transferred (one float loaded and stored), a theoretical lower bound on the kernel's runtime can consequently be computed as
$$
\frac{32\pi*10^6 \mathrm{B}}{256 * 10^9 \frac{\mathrm{B}}{\mathrm{s}}}=\frac{\pi}{8}10^{-3}\mathrm{s}.
$$

Seeing that this lower bound evaluates to a runtime of approximately 393 µs, our profiled runtime is very good. Looking at Nsight Compute's roofline chart, we see our kernel sitting clearly memory bound almost at the roofline, with an arithmetic intensity of 4.63 FLOP/B and a compute throughput of almost 1 TFLOPS. This discrepancy shows that our earlier theoretical estimate was only an approximation. Nsight Compute's measurements reflect the operations and memory traffic of the compiled kernel rather than those we counted by hand. In the below table we summarize each of the steps we took to arrive at our final version with their overall kernel runtime and relative performance improvement.

| Variant                 | Runtime (µs) | Reduction (%) |
| ----------------------- | ------------ | ------------- |
| Naive                   | 3528         | -             |
| Explicit multiplication | 1102         | 69            |
| Hoisted radius check    | 763          | 31            |
| Bounding square         | 560          | 27            |
| Vectorized              | 524          | 6             |
| Reduced memory access   | 408          | 22            |


### Marching Squares Kernel

#### Overview

We next take a look at the [marching squares](src/compute/kernels/marching_squares.cu) kernel. This kernel uses the [marching squares algorithm](https://en.wikipedia.org/wiki/Marching_squares) to compute the contour lines of an underlying height grid. For the specified threshold values, this algorithm approximates respective contours with linear segments at a resolution of the grid's underlying dimensions.

This algorithm may thereby be viewed as some sort of amalgamation between a convolution and an (unstable) filter kernel. Like a convolution kernel, we stride over the whole input grid with $2\times2$ subgrids. However, unlike a convolution and behaving more like a filter, each subgrid produces a variably sized output. For every four adjacent grid vertices (a $2\times2$ subgrid) and a single threshold, the algorithm may produce zero, one or two contour segments.

A simple solution to deal with this dynamic output would be to have each $2\times2$ subgrid always produce its maximal amount of output segments and setting the unused ones to `NaN` or some values outside the displayed area. However, that would take away a lot from the challenges for optimizing the kernel. The objective we are actually interested in. Plus, a kernel that yields only as many output segments as actually required gives us a nice, more generally usable solution and takes away work from the shader that eventually renders them.

```C++
__global__ void marching_squares(float const* heights, Size grid_size, int* contour_value_count, float* contours, float threshold) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < grid_size.width - 1 && row < grid_size.height - 1) {
        SubGrid subgrid = load_subgrid(heights, grid_size, col, row);

        // Computes the contour type for the subgrid 
        // heights[col, row] - heights[col + 1, row + 1].
        // Produces a number between 0 and 15
        int type = compute_type(subgrid, threshold);
        // Either 0, 1, or 2
        int segment_count = count_for_type(type);

        cuda::atomic_ref<int, cuda::thread_scope_device> contour_value_count_ref(*contour_value_count);
        int offset = contour_value_count_ref.fetch_add(segment_count * 4, cuda::memory_order_relaxed);

        // Compute and store either no segment, 
        // one segment consisting of (x1, y1), (x2, y2) 
        // or two such segments to contours
    }
}
```

As before, we start with a naive implementation and will work our way towards a more performant version. We choose a block size of $16\times16$, use no shared memory and write the segments to an output buffer indexed via a device-scoped atomic counter that tracks the number of scalar output values. For the underlying height grid we again choose a size of $8000\times4000$. For now, we compute the contours only for a single threshold sitting right at the middle of the height grid's range. Later on, we will also look at the case of multiple thresholds.

#### Analysis

Let us first take a look at the computational intensity of this kernel. Since the amount of computational steps, bytes loaded and stored differs with the amount of output segments, we present the different cases in the below table.

| **Case**         | 0 Segments | 1 Segment | 2 Segments |
| ---------------- | ---------- | --------- | ---------- |
| **Bytes Loaded** | 20         | 20        | 20         |
| **Bytes Stored** | 4          | 20        | 36         |
| **FLOPs**        | 0          | 8         | 38         |
| **OP/B**         | 0          | 0.2       | 0.68       |

We can already notice that in this naive implementation, the kernel is heavily memory bound, even more so than the previous smoothstep kernel. Further we see that the computational intensity differs by quite a bit across the various output conditions.

To get a feeling for the frequency, with which we can expect each of these three cases to arise, we launch the kernel across several grid configurations. In particular, we look into three categories: regular sinusoidal grids, and two groups of Perlin noise, one producing more gentle and the other more turbulent terrain. For each category we vary parameters and contour threshold across their respective ranges. The distribution of the three output cases for each category is displayed in the below table. We observe a very skewed distribution as for most $2\times2$ subgrids, we won't be doing any computation at all. Only rarely, a subgrid will actually produce an output segment and two are rarer still.

| **Case**                   | 0 Segments | 1 Segment | 2 Segments |
| -------------------------- | ---------- | --------- | ---------- |
| **Sinusoidal**             | ~99.9%     | ~0.1%     | ~0%        |
| **Gentle Perlin Noise**    | ~99.8%     | ~0.2%     | ~0%        |
| **Turbulent Perlin Noise** | ~99.1%     | ~0.89%    | ~0.01%     |

As a grid initialized with gentle Perlin noise appears to be the more interesting and realistic case, we will use it to continue with our profiling (parameters: octaves 4, frequency 6, persistence 0.5). Since this grid produces most of its segments close to the midpoint of its height range, we choose this midpoint as our contour threshold. At this threshold, our selected initialization produces only 0 and 1, but no 2, output segments per $2\times2$ subgrid. The actual counts are shown below.

| **Case**       | 0 Segments | 1 Segment | 2 Segments |
| -------------- | ---------- | --------- | ---------- |
| **Count**      | 31,853,485 | 134,516   | 0          |
| **Occurrence** | ~99.6%     | ~0.4%     | 0%         |

#### Using vector stores for the output
In its basic implementation, the performance of our kernel is quite poor. We measure a runtime of 2.76 ms on our $8000\times4000$ profiling grid. However there is an easy win waiting for us. Each contour segment consists of two $(x, y)$ start and end points, which maps naturally onto a single `float4` vector.

```C++
__global__ void marching_squares(float const* heights, Size grid_size, int* contour_count, float4* contours, float threshold)
```

We already observed in the smoothstep kernel how vector stores and loads improved performance. In this kernel the gain is even more pronounced. Switching to `float4` stores, we observe a speedup by a factor of 1.33. The runtime is now at 2.07 ms and the number of executed instructions decreased by 37% (although we suspect that some of these gains may also be attributed to how we restructured the code to allow for this modification).

#### Splitting the kernel into two
With the naive single-kernel implementation, we can expect warp divergence to be quite high. From the distribution measured above, on average only around one thread in every 250 will perform any floating-point computation. A single active thread is enough to prevent its entire warp from retiring early. Precious execution time that could otherwise be spent computing contours on a different region of the height grid.

Instead of computing the contours in a single pass, we can split our kernel into two. The first kernel checks each grid cell for the presence of a contour segment. If one is found, the cell's grid indices are written to a temporary buffer (already using `int2`'s for convenience). As in the basic implementation, the output slot is determined by atomically increasing a global counter. The second kernel then reads each such index pair, computes the corresponding contour segment, and writes it, reusing the same index, to the final contour buffer. We could also store only the cell’s linearized grid index as a single integer from the first kernel. However, the second kernel would then need to recover the row and column indices, which requires an integer division that we would rather avoid.

```C++
__global__ void marching_squares_phase_1(float const* heights, Size grid_size, int* contour_count, int2* coordinates, float threshold) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < grid_size.width - 1 && row < grid_size.height - 1) {
        SubGrid subgrid = load_subgrid(heights, grid_size, col, row);
        int type = compute_type(subgrid, threshold);
        int segment_count = count_for_type(type);  

        cuda::atomic_ref<int, cuda::thread_scope_device> contour_count_ref(*contour_count);
        int offset = contour_count_ref.fetch_add(segment_count, cuda::memory_order_relaxed);

        if (segment_count > 0) {
            coordinates[offset] = int2(col, row);
        }
        if (segment_count > 1) {
            // The second segment will be computed together 
            // with the first by the second kernel. 
            // Hence the index is marked off.
            coordinates[offset + 1] = int2(-1, -1);
        }
    }
}

__global__ void marching_squares_phase_2(float const* heights, Size grid_size, int const* contour_count, int2 const* coordinates, float threshold, float4* contours) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < *contour_count) {
        int2 value = coordinates[index];
        int col = value.x;
        int row = value.y;

        // A negative x-coordinate marks a second segment
        // that is generated by the previous thread.
        if (col != -1) {
            SubGrid subgrid = load_subgrid(heights, grid_size, col, row);
            int type = compute_type(subgrid, threshold);

            // Compute the 1 or 2 contour segments for this type
            //  and store them at contours[index] 
            // (and contours[index + 1] respectively)
        }
    }
}
```

To profile this modified setup, we need to switch to CUDA events as opposed to Nsight Compute, the profiling tool that we've used so far. Because Nsight Compute introduces profiling overhead by its nature and may rerun kernels several times to gather all information, the reported runtimes may differ from those measured with CUDA events and are typically higher. The conclusions we make about the relative kernel runtimes, however, hold regardless.

CUDA events allow us to capture timings across multiple kernel launches, including any potential memory transfers in between. In our setup, we run each variant 100 times after 5 warmup iterations and report the average runtime. Using this method, we observe an average runtime of 1429 µs for the current version of our kernel before applying any division.

In a first attempt, we launch the second kernel with the same amount of threads as there are potential contour lines, leaving the counter resident on the GPU. This yields a degraded runtime of 1551 µs, of which 1137 µs are attributed to first and 414 µs to the second kernel. In this implementation, the second kernel wastes the vast majority of its threads doing nothing.

We can do better by copying the counter back to the CPU and launching the second kernel with exactly as many threads as there are segments to compute. Despite the added cost of the device-to-host transfer, the much smaller dispatch more than compensates, improving the overall runtime down to 1194 µs, now using 17 µs and 40 µs for the memory transfer and second kernel respectively.

```C++
marching_squares_phase_1<<<...>>>(heights, grid_size, contour_count, coordinates, threshold);

int contour_count_h;
cudaMemcpy(&contour_count_h, contour_count, sizeof(int), cudaMemcpyDeviceToHost);


marching_squares_phase_2<<<...>>>(heights, grid_size, contour_count_h, coordinates, threshold, contours);
```

Splitting the kernel into two parts, we improved our algorithm by approximately 16%. In theory, we are still left with warp divergence in the second kernel due the differences for the one and two output segment cases. But the two segment case is so rare, that we can essentially neglect it. Subdividing the kernel further would introduce additional overhead that outweighs any gains from eliminating this remaining divergence.

#### Storing the height grid in shared memory
As a further benefit of having two kernels, we can now profile each stage separately. An obvious next optimization that comes to mind is shared memory. With computation happening in $2\times2$ subgrids, most data is actually reused by other threads. If we store square blocks of size $n$ in shared memory, the amount of repeatedly loaded halo cells is $2(n-1)+3=2n+1$. A quite low number when compared to $n^2$, the amount of inner cells that are loaded only once. Comparing this to $4n^2$, the number of cells loaded in the naive solution, shared memory allows us to in theory load only one quarter of the original amount as $n\to\infty$.

```C++
int col = blockIdx.x * BLOCK_DIM_X + threadIdx.x;
int row = blockIdx.y * BLOCK_DIM_Y + threadIdx.y;

// Halo values are loaded directly from global memory
__shared__ float heights_s[BLOCK_DIM_Y][BLOCK_DIM_X];

if (col < grid_size.width && row < grid_size.height) {
    heights_s[threadIdx.y][threadIdx.x] = heights[row * grid_size.width + col];
}
__syncthreads();

// Continue..
```

Trying it in practice, we however observe a performance far worse than that of the naive implementation loading all data separately. The use of shared memory now adds additional instructions and more importantly, barriers to our kernel. We can now observe way more warps stalling and doing nothing, waiting for their block's data to arrive. The amount of data reuse is too small for use to benefit from it. Especially if we consider that most memory accesses are likely served from L2 or even L1 cache.

#### Packing the grid values for reuse
We noticed in the previous section, that even though there are repeated loads, the first kernel still exhibits a favorable memory access pattern. The second kernel, in its current form, unfortunately doesn't. Memory accesses can be scattered around the height grid and two neighboring threads need not access neighboring data. As a result, the L2 cache hit rate is only around 57.5%. This is a relatively low value considering that each thread accesses two pairs of consecutive data elements by default.

Since data reuse is limited, we apply the same idea as previously and store the corresponding $2\times2$ subgrid values as packed `float4` vectors in the first kernel, alongside the grid indices. The second kernel can then load these packed values instead of fetching the data from the height grid directly. While this introduces an additional store instruction in the first kernel, it allows the second kernel to access its input in a fully coalesced manner.

```C++
__global__ void marching_squares_phase_1(float const* heights, Size grid_size, int* contour_count, int2* coordinates, float4* subgrid_heights, float threshold) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < grid_size.width - 1 && row < grid_size.height - 1) {
        SubGrid subgrid = load_subgrid(heights, grid_size, col, row);
        int type = compute_type(subgrid, threshold);
        int segment_count = count_for_type(type);

        cuda::atomic_ref<int, cuda::thread_scope_device> contour_count_ref(*contour_count);
        int offset = contour_count_ref.fetch_add(segment_count, cuda::memory_order_relaxed);

        if (segment_count > 0) {
            coordinates[offset] = int2(col, row);
            // The 4 subgrid values
            subgrid_heights[offset] = pack(subgrid);
        }
        if (segment_count > 1) {
            coordinates[offset + 1] = int2(-1, -1);
        }
    }
}

__global__ void marching_squares_phase_2(int contour_count, int2 const* coordinates, float4 const* subgrid_heights, float threshold, float4* contours) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < contour_count) {
        int2 value = coordinates[index];
        int col = value.x;
        int row = value.y;

        if (col != -1) {
            float4 packed_subgrid = subgrid_heights[index];
            int type = compute_type(packed_subgrid, threshold);

            // Continue..
        }
    }
}
```

Although this optimization reduces the runtime of the second kernel by approximately 39%, it also increases the runtime of the first kernel slightly by about 1% due to the additional store instructions. Overall, the combined runtime appears to be marginally lower than that of the original implementation. However, we were unable to reproduce this improvement consistently across repeated measurements. The second kernel is simply too small when compared to the first, in order to make a difference here. Given the additional implementation complexity and the lack of a reproducible speedup, we decided not to pursue this optimization further.

#### Reducing pressure on the atomic counter
We need to find another way to optimize our kernels. Looking at the warp stall statistics of the first, we observe a very high cycle count for *Stall Long Scoreboard*. This metric indicates that our kernel's warps spend a lot of their lifetime waiting on long-latency memory operations to complete. This might be due to the kernel's general memory requirements but inspecting the code we actually see another culprit. Every thread increments the global atomic counter for the number of output segments, regardless of whether its increment is zero or not.

All of these operations put a lot of pressure on the counter. After modifying the code to only increment the counter when a segment is actually produced, we find the above stall metric almost halved, now averaging only around 6 cycles. This is an optimization nvcc cannot perform itself, as it may not elide atomic operations. Consequently, the amount of *Eligible Warps Per Scheduler* is now increased by 73% to 1.53. Overall, reducing the contention on the atomic variable decreased the runtime of the first kernel from 1.81 ms to 1.06 ms.

```C++
int segment_count = count_for_type(type);

if (segment_count > 0) {
    cuda::atomic_ref<int, cuda::thread_scope_device> contour_count_ref(*contour_count);
    int offset = contour_count_ref.fetch_add(segment_count, cuda::memory_order_relaxed);
}
```

We could try to go further, aggregate the increments per warp or block, and only write to the counter once per each grouping. Instead of having each thread increment the global counter, only one per warp, respectively block, would be responsible to increment it with the collected aggregate. In the former approach, the individual increments can be shuffled via the cooperative groups API to a leading thread responsible for incrementing the global counter. In the latter, a block scoped atomic counter is incremented by each of the block's threads before one dedicated one writes it to the global counter. Each individual thread within this grouping is then able to determine its global index by looking at the just obtained index and their local index within it.

Under certain circumstances, nvcc would be able to perform the warp wide aggregation by itself. Our situation however, where each thread may increment by a value 0, 1 or 2, is too eclectic for the compiler to implement it. Regardless, these approaches do not give us much at this point in time. As we saw in the analysis of the marching squares algorithm, the case of having a grid vertex produce a contour segment is very rare. So rare that any form of synchronization actually degrades performance. We illustrate the runtime of the kernels in the mentioned variants in the below table.

| **Method**  | Write from each thread | Write if > 0 | Write once per warp | Write once per block |
| ----------- | ---------------------- | ------------ | ------------------- | -------------------- |
| **Runtime** | 1.81 ms                | 1.06 ms      | 1.49 ms             | 1.25 ms              |

#### Computing contours at multiple thresholds
So far we computed our height grid's contours only at a single threshold. For our specific use case in this application we, however, want to be able to display contours at various thresholds. Our so far chosen threshold sits at the midpoint of the grid's height range. The further we move it threshold towards the range's bounds, the fewer output segments it would produce. The below table illustrates the amount of output segments for evenly spaced thresholds across the grid's height range. To profile the marching squares kernel in the multi-threshold setup from here on, we will use 100 values, evenly spaced across the same range from $-\frac{3}{5}$ to $\frac{3}{5}$. This produces a total of 4,753,797 output segments.

| **Position** | $-\frac{3}{5}$ | $-\frac{2}{5}$ | $-\frac{1}{5}$ | $0$     | $\frac{1}{5}$ | $\frac{2}{5}$ | $\frac{3}{5}$ |
| ------------ | -------------- | -------------- | -------------- | ------- | ------------- | ------------- | ------------- |
| **Segments** | 662            | 7,540          | 69,385         | 134,516 | 76,476        | 10,044        | 86            |

We also make three assumptions: we will only know the exact number of thresholds at runtime, meaning we would need to define an upper bound if wanted to place them into constant memory, allow for unevenly spaced thresholds, and do not care about the ordering in which the output segments are produced. The latter one opens the way for optimizations as it loosens restrictions on how data should be stored. In our implementation, we also choose not to retain which threshold produced each final segment. This prevents us from differentiating between contours at different thresholds, e.g. if we wanted to color each one differently.

#### Enabling parallelism across multiple thresholds
What currently blocks us from computing contours at multiple thresholds in parallel is that we do not know which grid indices correspond to which threshold. From the first kernel, we currently save only the indices at which the second kernel then computes the contour segments. If we were to compute multiple thresholds in parallel, the second kernel would not know which threshold to use. Without any modifications, this leaves us naively computing the segments sequentially, one threshold after another.

However, that would leave several optimizations underutilized. To enable them, we can create a second temporary output buffer, to which the first kernel stores the corresponding thresholds. Each element of this buffer corresponds to an index pair in the temporary vertex buffer. The second kernel then loads both buffers and is consequently able to identify a grid index pair together with its threshold.

```C++
__global__ void marching_squares_phase_1(float const* heights, Size grid_size, int* contour_count, int2* coordinates, float* subgrid_thresholds, float threshold) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < grid_size.width - 1 && row < grid_size.height - 1) {
        SubGrid subgrid = load_subgrid(heights, grid_size, col, row);
        int type = compute_type(subgrid, threshold);
        int segment_count = count_for_type(type);

        if (segment_count > 0) {
            cuda::atomic_ref<int, cuda::thread_scope_device> contour_count_ref(*contour_count);
            int offset = contour_count_ref.fetch_add(segment_count, cuda::memory_order_relaxed);

            coordinates[offset] = int2(col, row);
            subgrid_thresholds[offset] = threshold;

            if (segment_count > 1) {
                coordinates[offset + 1] = int2(-1, -1);
            }
        }
    }
}

__global__ void marching_squares_phase_2(float const* heights, Size grid_size, int contour_count, int2 const* coordinates, float const* subgrid_thresholds, float4* contours) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < contour_count) {
        int2 value = coordinates[index];
        int col = value.x;
        int row = value.y;

        if (col != -1) {
            float threshold = subgrid_thresholds[index];
            SubGrid subgrid = load_subgrid(heights, grid_size, col, row);
            int type = compute_type(subgrid, threshold);

            // Continue..
        }
    }
}
```

#### Three approaches to execute in parallel
With this tweak, we are now able to launch the first kernel in batch for all thresholds simultaneously. Each launch is assigned to a dedicated CUDA stream, in which it can execute independently of the others. The second kernel then only has to run once, after all previous ones have finished.

```C++
for (int i = 0; i < threshold_count; ++i) {
    marching_squares_phase_1<<<..., streams[i]>>>(heights, grid_size, contour_count, coordinates, subgrid_thresholds, thresholds[i]);
}
cudaDeviceSynchronize();

int contour_count_h;
cudaMemcpy(&contour_count_h, contour_count, sizeof(int), cudaMemcpyDeviceToHost);

marching_squares_phase_2<<<...>>>(heights, grid_size, contour_count_h, coordinates, subgrid_thresholds, contours);
```

In this approach, we are still left with quite some kernel launches. Even though we run them in parallel, the first kernel is launched 100 times. CUDA Graphs are a great way to speed up such a situation. Instead of launching each kernel individually from the host, we can capture the sequence of launches into a graph once and then replay this graph as a single unit for every subsequent call. This removes most of the CPU-side launch overhead that would otherwise accumulate.

To avoid these multiple launches of the first kernel altogether, we can instead map the thresholds onto a third dimension after the grid's height and width. This allows us to launch a single kernel on a 3-dimensional grid and compute all thresholds together. This approach brings us back to two kernel launches, but now computing contours at multiple thresholds in the first one. The first, 3-dimensional kernel populates the temporary index and threshold buffers. The second kernel then computes the segments from these.

As a consequence of this three-dimensional mapping, we can no longer pass the thresholds directly as the first kernel's parameters. Since we only know the threshold count at runtime, this leaves us no option other than passing a pointer to them instead. Each thread then has to load its threshold dynamically at runtime from its index along the z-axis.

```C++
__global__ void marching_squares_phase_1(float const* heights, Size grid_size, int* contour_count, int2* coordinates, float* subgrid_thresholds, float const* thresholds, int threshold_count) {
    int layer = blockIdx.z * blockDim.z + threadIdx.z;

    if (layer < threshold_count) {
        float threshold = thresholds[layer];

        // Continue..  
    }
}
```

#### Comparing the four approaches
The performance of these four approaches is reported in the below table. In the first row, we show the total runtime for each, spanning both the (potentially multiple) first kernel launches and the second kernel launch. As before, these times were measured using CUDA events and calculated as the average of 100 runs after 5 warmup iterations. In the second and third rows, we report the runtime of the first and second kernels respectively. These were measured using Nsight Compute, with a single threshold placed at the height grid's midpoint.

| Variant           | Naive    | Batched  | Batched & CUDA Graphs | 3D Grid    |
| ----------------- | -------- | -------- | --------------------- | ---------- |
| **Combined (100 thresholds, CUDA events)**      | 86.86 ms | 95.61 ms | 87.26 ms              | 92.44 ms   |
| **First Kernel (1 threshold, Nsight Compute)**  | 1.04 ms  | 1.05 ms  | 1.05 ms               | 1.13 ms    |
| **Second Kernel (1 threshold, Nsight Compute)** | 0.05 ms  | 0.05 ms  | 0.05 ms               | 0.05 ms    |

We observe an unexpected behavior. None of the modifications were actually able to improve the overall runtime. The processors are already busy computing contours at one threshold, so our effort to parallelize could not help much. Also, the batched version's additional DRAM storage and stream management overhead actually seem to be working against it. Using CUDA Graphs, we are able to make up for this somewhat, but still do not manage to outperform the naive implementation.

The three-dimensional approach performs rather poorly as well. Even though it only launches two kernels, its runtime is worse than the naive version's. In particular, the added dependency of indirectly loading the thresholds seems to hurt its first kernel's performance. We also notice that the second kernel plays only a minor role in the overall setup.

#### Coarsening the threads along the thresholds
Despite its initially weak performance, we decide to continue with the three-dimensional implementation. Its shape gives us a huge advantage: the ability to coarsen threads along the new threshold dimension. Instead of having one thread compute segments at one $2\times2$ subgrid and one threshold, we can instead let it do so for multiple thresholds. The subgrid stays fixed and doesn't need to be reloaded throughout these repeated computations. This saves us costly time, as the thread now only stalls once waiting for its data from global memory, instead of stalling for each threshold as in the un-coarsened version.

```C++
int layer = (blockIdx.z * blockDim.z + threadIdx.z) * COARSE_FACTOR;

for (int c = 0; c < COARSE_FACTOR; ++c) {
    if (layer + c >= threshold_count) {
        return;
    }

    float threshold = thresholds[layer + c];

    // Continue..
}
```

We measured the runtimes of the three-dimensional kernel at various coarsening factors and show them in the below table. As before, we report them once for the individual kernels, measured with Nsight Compute, and once for their combination, measured with CUDA events. Starting already at a coarsening factor of two, we can see the runtime rapidly declining, outperforming the naive version considerably. Please note that in the previous section, we profiled the individual kernels at only one threshold, while now we do so for all 100, hence the difference in runtime.

| Coarse Factor     | 1         | 2        | 4        | 8        | 16       | 32       | 64       | 128      |
| ----------------- | --------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| **Combined (CUDA events)**      | 93.45 ms  | 59.58 ms | 42.61 ms | 34.78 ms | 30.81 ms | 28.31 ms | 27.37 ms | 25.69 ms |
| **First Kernel (Nsight Compute)**  | 137.02 ms | 84.28 ms | 61.28 ms | 53.98 ms | 50.56 ms | 48.98 ms | 47.84 ms | 47.29 ms |
| **Second Kernel (Nsight Compute)** | 1.73 ms   | 1.36 ms  | 1.13 ms  | 1.08 ms  | 1.07 ms  | 1.06 ms  | 1.06 ms  | 1.05 ms  |

The reduced memory traffic really paid off. As a consequence, we observe the L1 and L2 cache hit rates increase by 28% and 95% respectively, when comparing the un-coarsened version to the version coarsened by a factor of 128. Interestingly, the second kernel also improved. This kernel loads the height grid's data from the grid indices that were stored in the temporary buffer by the first kernel. The closer together the data represented by these indices are, the more efficiently they can be loaded from global memory. As a consequence of the thread blocks staying longer in one area of the height grid, we get less variance in the resulting index buffer. An increased memory throughput of approximately 64% across all layers confirms this optimized pattern.

Seeing that the runtime improvement slowly starts to plateau after a coarsening factor of 16, we settle on 32 from here on. This value seems to be a good middle ground: large enough to give a measurable runtime improvement, yet small enough to avoid the diminishing returns and excessive coarsening seen at higher factors.

#### Storing the thresholds in shared memory
Now that each thread is fetching multiple thresholds from global memory, we can take our second attempt at making use of shared memory. While in our previous, unsuccessful attempt with the height grid each vertex was only shared by a maximum of four threads, we now share thresholds across the full block. With our current grid configuration of $16\times16\times1$ for launching the first kernel, the reuse is quite high.

We see two ways of utilizing shared memory in this situation. For one, we could collectively load the full 32 (our chosen coarse factor) thresholds into shared memory upon kernel entry and then synchronize. Alternatively, we could also have a single thread load the respective threshold value into shared memory at each iteration step. With 32 required barriers, the synchronization effort would be rather extensive in this latter approach.

```C++
int layer = (blockIdx.z * blockDim.z + threadIdx.z) * COARSE_FACTOR;
int thread_id = threadIdx.y * blockDim.x + threadIdx.x;

__shared__ float threshold_s[2];

for (int c = 0; c < COARSE_FACTOR; ++c) {
    if (layer + c >= threshold_count) {
        return;
    }

    // We alternate the index to avoid write after read hazards
    int index = c % 2;

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        threshold_s[index] = thresholds[layer + c];
    }

    __syncthreads();

    // Compute contour at threshold_s[index]
}
```

We profiled both approaches. In the latter, we observe a very negligible performance increase. Intuitively this makes sense. Storing the threshold in shared memory does not give us much. We introduced additional barriers for a value that very likely was already inside L1 Cache. Looking at Nsight Compute, we see behavior consistent with this suspicion. L1 Cache hit rate decreased by 18% while warps now spend on average 2.5 cycles per instruction stalling at a barrier. Given these new constraints, the slight performance increase is actually rather remarkable.

Moving on to the former approach, we see a more pronounced effect. The first kernel's runtime is decreased by 7 ms, down to now only 41.94 ms. While the kernel requires more effort initially, loading all thresholds and synchronizing, it is able to move very fast from there on. After the initial collective load, no more thresholds need to be loaded from global memory. As before, we observe a decreased L1 Cache hit rate as a side effect. Seeing the benefit this approach has on the first kernel's performance, we use it from here on.

```C++
int layer = (blockIdx.z * BLOCK_DIM_Z + threadIdx.z) * COARSE_FACTOR;
int thread_id = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
int stride = BLOCK_DIM_Z * blockDim.y * blockDim.x;
int layer_count = min(BLOCK_DIM_Z * COARSE_FACTOR, threshold_count - layer);

__shared__ float thresholds_s[BLOCK_DIM_Z][COARSE_FACTOR];

for (int i = thread_id; i < layer_count; i += stride) {
    thresholds_s[i / COARSE_FACTOR][i % COARSE_FACTOR] = thresholds[layer + i];
}
__syncthreads();

// Compute contours at thresholds_s
```

To be fair, we should add that only about half of this 7 ms performance gain can be attributed to the use of shared memory itself. The other half came from hoisting the repeated `if (layer + c >= threshold_count)` check out of the loop into a single computation of `layer_count`. While it only became necessary in the shared memory version, this single change would also have helped the basic implementation.

#### Unrolling the loop over the thresholds
Having a loop over the individual thresholds inside our kernel, we can hint the compiler to unroll it. An unrolled loop exposes more independent instructions to the scheduler, which it can then work with to exploit instruction level parallelism and better hide latencies. While the pragma to unroll a loop is only a hint to the compiler, it seems to have followed through with it in our case. When hinting to unroll in powers of 2 for the range 1 to 32, we can see actual performance differences which we show in the below table.


```C++
int layer = (blockIdx.z * BLOCK_DIM_Z + threadIdx.z) * COARSE_FACTOR;

SubGrid subgrid = load_subgrid(heights, grid_size, col, row);

// Load thresholds into thresholds_s

int compute_layers = min(COARSE_FACTOR, threshold_count - layer);

#pragma unroll 16
for (int i = 0; i < compute_layers; ++i) {
    float threshold = thresholds_s[threadIdx.z][i];

    int type = compute_type(subgrid, threshold);
    int segment_count = count_for_type(type);

    if (segment_count > 0) {
        // Continue..
    }
}
```

We observe a noticeable decrease in runtime up to a factor of 16, after which the runtime increases again. At a factor of 32, we basically eliminate the loop entirely for all but the boundary blocks, given the chosen coarse factor of 32. At the most performant unroll factor of 16, we leave the loop at only two iterations. Interestingly, the compiler seems to have already unrolled the loop by a factor of two on its own, as explicitly setting it to 1, i.e. having no unrolling at all, shows a noticeable performance degradation.

| Unroll Factor | None     | 1        | 2        | 4        | 8        | 16       | 32       |
| ------------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| **Runtime**   | 41.94 ms | 43.12 ms | 41.65 ms | 38.55 ms | 36.99 ms | 36.81 ms | 38.40 ms |

Looking at the profiler output, we have another interesting observation. Irrespective of the chosen unroll factor, the threads use the same amount of registers. The compiler seems to be pretty good at reusing them. However this also suggests the performance increase doesn't really come from an improved instruction level parallelism. We expect such a tight register reuse to introduce data hazards which directly would contradict an increased parallelism. Instead the performance gain appears to be coming from a reduced control overhead for the loop. We see issued instructions decreased by 15% and branch instructions decreased by 23% when comparing the kernel with unroll factor 16 against the base implementation with no compiler hint at all.

#### Privatizing the output data
A common approach to optimize a filter type kernel is to apply block privatization. In the chapter about reducing the pressure on the atomic counter, we already quickly tried to aggregate the counts locally per block before writing them once to the global counter. Back then, the rarity of actually containing an output segment prevented this method from being useful to. Now that we coarsen the kernel along the thresholds, we should get the coarse factor times more output segments. With out chosen factor of 32, each block is on average responsible for about 12 segments. This is still a pretty sparse distribution, but at least better than before.

However, differently to before, where we had only one threshold per block, we now can no longer keep any potential output data in registers until their slot in global memory is determined. Each height grid vertex could in theory produce up to $2f_c$, where $f_c$ is the coarse factor, output segments. Even though we know that in practice, this number will almost always be way lower than this, we still have no guarantee for it. To not run out of registers and having to spill them to DRAM, we can instead buffer the output data in shared memory. As a neat benefit of this modification, we will collectively write the output in a coalesced manner at the kernel's end and do not scatter it in unconnected locations as we do in the previous version.

```C++
SubGrid subgrid = load_subgrid(heights, grid_size, col, row);

// Store thresholds into shared memory

__shared__ int2 coordinates_s[BUFFER_SIZE];
__shared__ float subgrid_thresholds_s[BUFFER_SIZE];
__shared__ int block_count_s;
__shared__ int global_offset_s;

if (thread_id == 0) {
    block_count_s = 0;
}

__syncthreads();

#pragma unroll 16
for (int i = 0; i < compute_layers; ++i) {
    float threshold = thresholds_s[threadIdx.z][i];

    int type = compute_type(subgrid, threshold);
    int segment_count = count_for_type(type);

    if (segment_count > 0) {
        cuda::atomic_ref<int, cuda::thread_scope_block> block_count_ref(block_count_s);
        int block_offset = block_count_ref.fetch_add(segment_count, cuda::memory_order_relaxed);

        coordinates_s[block_offset] = int2(col, row);
        subgrid_thresholds_s[block_offset] = threshold;

        if (segment_count > 1) {
            coordinates_s[block_offset + 1] = int2(-1, -1);
        }
    }
}

__syncthreads();

if (thread_id == 0 && block_count_s > 0) {
    cuda::atomic_ref<int, cuda::thread_scope_device> global_count_ref(*count);
    global_offset_s = global_count_ref.fetch_add(block_count_s, cuda::memory_order_relaxed);
}

__syncthreads();

for (int i = thread_id; i < block_count_s; i += stride) {
    coordinates[global_offset_s + i] = coordinates_s[i];
    subgrid_thresholds[global_offset_s + i] = subgrid_thresholds_s[i];
}
```

While we now saved registers from being overused, we shifted the problem to shared memory. On our profiling GPU, shared memory is limited to 100 KiB per streaming multiprocessor. To get full occupancy with the possible 1536 threads, we cannot use more than 66.7 bytes per thread. With our coarse factor of 32, this is way too little to potentially store all the potentially required data. We can nevertheless get a feeling for this method by simply limiting the buffer size to fit for now. With our profiling grid, no block produces more than 140 output segments. An amount, we easily can accommodate in shared memory.

Looking at the metrics, we can see the added benefits we expected confirmed. The utilization of the L2 atomic input path handling our global atomic (*lts__d_atomic_input_cycles_active.max.pct_of_peak_sustained_elapsed*) decreased from 15.1% to now only 0.4%. Warps stalled on long latency memory dependencies (*Stall Long Scoreboard*) for only 0.02 instead of 0.12 cycles per issued instruction. The average number of useful global store bytes per transferred 32 byte sector (*smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio*) increased from 13.96 to 26.83 due to improved coalescing.

Upon observing the kernel's runtime, we are however hit with a surprise. The privatized implementation performs about 1.3 ms worse than its baseline. The output segments are simply still too sparsely distributed for the kernel to benefit from this modification. With a utilization of only 15.1%, the L2 atomic input path was far from sustained saturation to begin with. And while long-scoreboard stalls decreased, they were only a minor issue. In comparison, the additional synchronization increased barrier stalls (*Stall Barrier*) by 0.57 cycles per issued instruction. Additionally, we now have an increase in issued instructions by 2%. We increased the overhead of our kernel to solve an issue that never had a big impact to start with. Even if we experimentally increase the coarse factor to the full 100 thresholds, we observe the same behavior, albeit now with a smaller remaining runtime difference.

#### Pruning threshold iterations
We need to look elsewhere to find a more meaningful improvement for our kernel. Taking a step back to analyze it, we find it to be very much compute bound. It currently has a compute throughput of 83% and a memory throughput of only 33%. Coarsening the kernel and applying the subsequent modifications, it switched from being memory bound to being compute bound. Let us therefore take a look at the algorithm itself. Currently we compute the type of a $2\times2$ subgrid, the amount segments it produces and compare this amount against zero for every individual threshold. This is quite a bit of wasted computation if we consider that each subgrid is very likely to produce segments at most a few thresholds and even likelier at none at all. If we require the thresholds to come sorted in ascending order, we can perform a small trick that however greatly reduces computations. We can compute the minima and maxima of a subgrid and only check the thresholds that are lying within.

```C++
SubGrid subgrid = load_subgrid(heights, grid_size, col, row);

float min = minimum(subgrid);
float max = maximum(subgrid);

#pragma unroll 16
for (int i = 0; i < compute_layers; ++i) {
    float threshold = thresholds_s[threadIdx.z][i];

    if (threshold < min || max <= threshold) {
      continue;
    }
    
    // Continue..
}
```

Trying it out we see the first kernel's runtime being more than halved to 16.65 ms. We made the algorithm way simpler by reducing its operations. Consequently we also have the number of issued instructions reduced by 60%. But we can do even better. Currently we are still looping over each threshold individually, fetch it from shared memory and only then potentially exit the respective loop iteration early. Instead we can let each thread determine the exact range with which it needs to iterate over the thresholds. With this approach we can greatly reduce the amount of iterations each thread performs. We experiment with two ways to compute this range within the current batch of thresholds, a naive linear search (potentially exiting early) and a binary search.

```C++
SubGrid subgrid = load_subgrid(heights, grid_size, col, row);

float min = minimum(subgrid);
float max = maximum(subgrid);

// Either compute the bounds via binary search
int lower = lower_bound(min, thresholds_s[threadIdx.z], compute_layers);
int upper = lower_bound(max, thresholds_s[threadIdx.z], compute_layers);
// Or find them via linear search
auto [lower, upper] = compute_bounds(min, max, thresholds_s[threadIdx.z], compute_layers);

// We no longer unroll the loop
for (int i = lower; i < upper; ++i) {
    float threshold = thresholds_s[threadIdx.z][i];

    // Continue..
}
```

Both approaches give us a very noticeable further decrease in runtime with the binary search version being the more performant of both. As before, we attribute most of these gains the decreased operations the algorithm has to perform. We illustrate the runtime of each version together we the number of issued instructions in the below table. Since we expect each thread to mostly perform less than one loop iteration on average now, there is also no need for pragma to unroll them. After decreasing the operations of this kernel, it is no longer as clearly compute bound. For the version using binary search to find the range, we observe a rather balanced compute throughput of 67% and a memory throughput of 52%.

| Variant                 | Full Loop     | Early Continue | Linear Pruning | Binary Pruning |
| ----------------------- | ------------- | -------------- | -------------- | -------------- |
| **Runtime**             | 36.81 ms      | 16.66 ms       | 13.57 ms       | 12.20 ms       |
| **Issued Instructions** | 2,577,154,302 | 1,019,366,501  | 793,036,861    | 670,953,713    |

#### Revisiting threshold coarsening and the z-dimension
While we are in theory still left with a loop over the, now pruned, thresholds, we expect them to be at most a single iterations for almost all threads. In order for a $2\times2$ subgrid to produce segments for two or more thresholds, they would either need to be very close together or the heights at this subgrid would need to be very steep. On our profiling grid, we in fact observe none such case. Previously we stopped at a coarse factor of 32 as we saw compute pipelines starting to become saturated. Now that we vastly decreased the number of instructions, we have a valid reason to visit this approach again. We can even go further and get rid of the 3rd threshold z-dimension altogether. This way each thread will be responsible for computing all potential contour segments at its respective subgrid.

```C++
int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
int stride = blockDim.x * blockDim.y;

// The size of this buffer is now defined at kernel launch time
// and equal to threshold_count
extern __shared__ float thresholds_s[];

for (int i = thread_id; i < threshold_count; i += stride) {
    thresholds_s[i] = thresholds[i];
}

__syncthreads();

if (col < grid_size.width - 1 && row < grid_size.height - 1) {
    // Load the subgrid and compute the loop range

    for (int i = lower; i < upper; ++i) {
        // Continue..
    }
}
```

Luckily we came back to this topic. The kernels runtime is now down at 6.63 ms, almost half of its baseline. As before we mostly attribute it to the by 61% decreased issued instructions, which are now at a value of 261,175,753. This decrease is also able to cover up quite a few things that turned worse now. We can observe warps stalling due pending memory requests. Possibly a result of an increased pressure on the atomic counter. We now get the same amount of requests in almost half the time as before. Indeed, the utilization at the L2 atomic input path (*lts__d_atomic_input_cycles_active.max.pct_of_peak_sustained_elapsed*) increased from 43% to 81%. Consistent with this increased pressure, we get almost one less eligible warp per scheduler. And more generally, streaming multiprocessors are now busy only 49% and memory pipelines only 34% of the time. Even though we made great improvements from before, our kernel now calls for further modifications.

#### Revisiting shared memory loads of the height grid
Le us take a look at another modification we tried previously and see if its position still holds. Most of the setup did not change that would affect the previous outcome for loading the height grid into shared memory at the first kernel's beginning. Only now, we have a barrier for loading the thresholds into shared memory that did not exist before. Previously, when one did not exist yet, the barrier we added degraded the kernel's performance a lot. Now that we have one anyways, we are interested if the situation differs.

A new profiling run unfortunately shows no improvement. With the current block configuration of $16\times16$ threads, we observe a few µs added to the baseline's runtime. Switching the configuration to $32\times16$, we manage to get them basically equivalent. Even with the new setup, we still cannot make shared memory useful for the height grid. As expected, DRAM traffic is down in this version and we find fewer warps stalling due to memory dependencies. But the time saved here, is now spent at the barrier instead. It seems the added overhead still outweighs the limited data reuse. Also the added complexity for these shared memory loads shows in an increase of 16% of issued instructions.

#### Revisiting block privatization
Having each thread now consider all thresholds after we removed the threshold dimension from the grid, we also improved output density per thread and block. In total all threads produce 4,753,797 output segments across all thresholds and approximately every seventh thread will be computing one of them. Since we previously determined output sparsity as the crucial factor inhibiting block privatization from being useful, we should take another look again. As before, we still cannot guarantee all output will fit in shared memory for all possible height grids. But since ours is nicely behaved, we can use it to get a feel for this optimization.

And indeed, it was worth it to look at this approach again. Using a block wide atomic and saving the output data in shared memory before writing it collectively to global memory decreased the first kernels runtime down to 4.39 ms. That is impressive, as in our previous attempt with even a coarse factor of 100, we did not see any benefit. As deciding factor, we make out the global atomic. In the previous attempt, the big loop over the threshold helped to spread out access to the global counter. Having decreased this loop vastly, we already observed an utilization of 81% at this atomics L2 input path. The aggregated collective write from each block, managed to bring it down to now only 3%. Pruning the loop clustered the atomic writes very tightly together and made it an actual bottleneck.

Consequently, we basically see warps no longer stalling and waiting on global memory requirements with now only 0.21 cycles per issued instruction compared to 7.53 in the pruned version. This vast decrease also amortizes the slight increase by 1.41 cycles in warps stalling due to the additional synchronization that we needed to add in. Alongside this reduction, we observe 1.37 additional eligible warps per scheduler on average and find the streaming multiprocessors busy around 77% of the time. Having coalesced writes instead of the scattered ones benefited us as well of course.

#### Generalizing block privatization

Despite this triumph. we should not forget that the current privatized implementation is not generalizable. So far we profited from the fact that our profiling height grid allows us to limit the amount of shared memory we need to reserve per block. Further, the output data we want to write comfortably fits inside shared memory. Theoretically, each thread could write 2 output segments for each threshold that the kernel visits. In the current approach, we write one `int2` for the grid indices and one `float` for the thresholds per output segment. Multiplying all this info together, each thread would in our case potentially need $2 * 100 * (4B + 8B)=2400B$. While this case is very exaggerated, we have no mechanism in place that would prevent it from happening in theory.

On our profiling GPU, that supports 1536 threads and 100 KiB per streaming multiprocessor, that is far more than we can provide.The simplest way around this limitation, that we can think of, is to only provide a shared memory buffer that we can support. Each block would then first fill this buffer and once it is full, write the remaining output data directly to global memory. Without limiting occupancy, we can provision $\frac{100KiB}{1536}\approx66.7B$ of shared memory per thread. Accounting for the thresholds which we already store in shared memory, plus any potential counters and flags, every thread could potentially write 5 output segments on average before the buffer becomes full. In most height grid instantiations, we expect this to be more than enough.

```C++
int block_offset;
// Atomically reserve space without increasing block_count_s beyond CAPACITY
if (try_reserve(block_count_s, segment_count, CAPACITY, block_offset)) {
    coordinates_s[block_offset] = int2(col, row);
    subgrid_thresholds_s[block_offset] = threshold;

    if (segment_count == 2) {
        coordinates_s[block_offset + 1] = int2(-1, -1);
    }
} else {
    cuda::atomic_ref<int, cuda::thread_scope_device> device_ref(*count);
    int global_offset = device_ref.fetch_add(segment_count, cuda::memory_order_relaxed);

    coordinates[global_offset] = int2(col, row);
    subgrid_thresholds[global_offset] = threshold;

    if (segment_count == 2) {
        coordinates[global_offset + 1] = int2(-1, -1);
    }
}
```

Trying it out, we find no runtime difference to the previous version. This makes sense. Compared to it, we only added in a branch that was never taken on our profiling grid. For a solution that is performant on a wide range of height grids, we would need to profile ours on such height grids as well. We can imagine that different approaches might be performant as well. E.g. one where the shared buffer is repeatedly written to global memory after every few threshold iterations. Or another one, in which each warp is given its own buffer, which it can flush to global memory whenever it becomes full.

Preempting the upcoming section, the actual approach we choose won't matter. We will find that with the next modification, the privatized approach will not benefit us any more. To ground this claim with actual measurements, we time the full two-kernel pipeline with CUDA events. Using the privatized first kernel, followed by a device-to-host transfer of the counter and the second kernel computing the contours from the intermediate data, we get a runtime of 3.7 ms averaged across 100 runs. 

#### Merging the kernels again
Instead of looking at shared memory, we can revisit a decision we made very early on in this profiling journey. Is the split of the algorithm into two distinct kernels, the first to compute the grid indices that produce an output segment and the second to then compute the respective contour segments, actually still necessary? Back then, the sparse output with only one threshold made this split a reasonable decision. Now that each thread considers all 100 thresholds and processes its pruned range, the setup is quite different. In the previous section we already established, that now approximately every 7th thread will produce an output segment. While this situation is still not very dense, it definitively is not as sparse as it was before.

```C++
SubGrid subgrid = load_subgrid(heights, grid_size, col, row);

for (int i = lower; i < upper; ++i) {
    float threshold = thresholds_s[i];

    int type = compute_type(subgrid, threshold);
    int segment_count = count_for_type(type);

    if (segment_count > 0) {
        cuda::atomic_ref<int, cuda::thread_scope_device> contour_count_ref(*contour_count);
        int offset = contour_count_ref.fetch_add(segment_count, cuda::memory_order_relaxed);
        
        // Directly compute and store one segment consisting of 
        // (x1, y1), (x2, y2) or two such segments to contours
    }
}
```

And indeed, measuring both approaches with CUDA events, we get a runtime of 4.8 ms for the split (not using privatization) and one of 3.7 ms for the combined kernel approach. The new setup completely changed the outcome with the combined approach outperforming the baseline by more than 1 ms. When profiling the kernels with Nsight Compute, we see the combined kernel outperforming even the first one of the split approach (without privatization) by 0.28 ms with a runtime of 6.35 ms. on top of that, we save lots of time by not needing to launch a second kernel or having to transfer the counter between GPU can CPU. Given the fact that the combined version has an increase in issued instructions by 39% compared to the the first in the split approach, this result is quite fascinating. This move managed to even amortize any potential benefit, we would get with a privatized first kernel implementation. Remember that we measured the combined runtime of this approach at 3.7 ms as well.

As the reason for this surprising behavior we make out the way the baseline kernel utilized memory pipelines. With its low compute throughput, the kernel's warps were mostly becoming eligible in bursts because of their common memory requirements. Once they had collectively fetched the thresholds from global memory, compute was small enough to have the threads fetch their respective $2\times2$ subgrid's heights and store back the determined indices at similar times. We can confirm that unresolved memory dependencies were a major limiting factor by looking at the profiling output in Nsight Compute. By far the most common warp stall reason of the first kernel in the split implementation was waiting on data from memory to arrive (*Stall Long Scoreboard*).

The combined kernel on the other hand manages to hide these memory latencies. Also here, warps stalling due to memory requirements is the most common reason. But we find their occurrence almost halved compared to the baseline. Even more important, the combined kernel now gets about 1.65 additional eligible warps per scheduler on average with the amount of active warps staying constant. This strengthens our theory of the first kernel in the split version becoming active in waves. By adding more work, we made the kernel performing better and managed to speed to up the whole algorithm considerably. With this new method, we now also see the kernel making more use of its hardware with streaming multiprocessors being busy 65% of the time. In a sense, we managed to get similar benefits from a privatized implementation via a different approach.

#### Fixing the atomic increment
With the decision to combine the kernels again, we made way for yet another optimization. Due to the way we decided to handle the rare two-segment count, we required them to occupy consecutive slots in the global array. In that implementation, a single thread of the second kernel would compute either output segments for convenience. To store the second segment, the thread needed the guarantee that the slot after the first segment in the global output array was available. While that was an implementation choice back then, the merged kernel lifts this requirement again. As the actual contour segments are already being computed in the kernel that reserves the slots in the global array, nothing prevents us from storing the output in a two-segment case in separate locations.

By actually doing so, i.e. storing the first and second segments separately, we can get the CUDA compiler to perform a neat optimization for us. By itself, the compiler is able to aggregate atomic writes per warp if the correct requirements are met. With increments being either one or two that is not the case. However, if we make use of the previously stated potential to separate the writes, we can make it happen. For each thread that at least writes one contour segment, we can always have it increment the atomic counter by one and write its, potentially only first, output to the obtained slot. To write the second segment in the cases where it applies, the counter would then be incremented again by one and the output written to the respective global location.

```C++
SubGrid subgrid = load_subgrid(heights, grid_size, col, row);

for (int i = lower; i < upper; ++i) {
    float threshold = thresholds_s[i];

    int type = compute_type(subgrid, threshold);
    int segment_count = count_for_type(type);

    if (segment_count > 0) {
        cuda::atomic_ref<int, cuda::thread_scope_device> segment_count_ref(*count);
        int global_count = segment_count_ref.fetch_add(1, cuda::memory_order_relaxed);

        // Compute and store the first contour segment

        if (segment_count > 1) {
            global_count = segment_count_ref.fetch_add(1, cuda::memory_order_relaxed);

            // Compute and store the second contour segment
        }
    }
}
```

With this trick, we basically aggregate the atomic write to the global counter per warp. Already previously we noted that the global atomic is quite a bottleneck in our code. Back then, we found the utilization at its L2 input path at 85% and saw quite a few warps stalling due to global memory requirements. With this modification applied, we see the former metric decreased to 18%. Not as good as we saw it when trying out block privatization, but still quite some jump. Similarly to that case, we basically no longer see any warps stalling due to global memory requirements with the additional benefit, that we did not add in any new barriers. With this approach, we see 1.25 more eligible warp available per scheduler on average. That is almost as good of an increase as before. Compared to the baseline, the kernels runtime is down by more than one millisecond to now 5.19 ms. 

#### Trying block privatization again
We might be tempted to go a step further and privatize the combined kernel as well. While the warp aggregated atomic writes already showed a huge improvement in the mentioned metrics, we could see if we can perform even better. Trying again an approach that limits shared memory to a fixed size, we however now do not get a performance increase. The utilization at the global atomics input path is again down to 3%, but the difference is not as severe as before. The warp aggregated atomic write already managed to dissolve the bottleneck at the global counter. And while we in a sense got the reduction in atomic writes for free, privatization only introduced additional barriers and instructions that hurt us now.

The number of cycles per issued instruction that warps now spend waiting at a barrier almost doubled from 1.58 to 3.03. Whereas the decrease in cycles spent waiting on global memory requirements is less pronounced with a jump from 0.46 down to 0.11. Additionally, the modifications we had to make on our code increased the number of issues instructions by 10%. These changes simply outweigh the benefit of the further reduced atomic pressure or coalesced writes, that got improved with the average number of useful global store bytes per transferred 32 byte sector being increased from 18 to 26.83. The overall runtime of this kernel variant sits at 5.68 ms.

#### Summary
This last try concludes the optimization journey for our marching squares kernel. Interestingly, the kernel's final shape is not too far off from the initial version right at the beginning. The initial conditions made it beneficial to split the kernel into two distinct phases that continued to benefit us even after we added it multiple contour threshold levels. Only after experimenting with different approaches, we then found a performant way that allowed us to fuse the two phases into a single kernel again. Notably, most assumptions that we expected to benefit this unstable filter and convolution like kernel, did not hold. Rather the small tweaks were, what helped us to ultimately arrive at a performant implementation.

While the initial version computed contours at a single threshold in 2.76 ms when measured with Nsight Compute, the final version computes all 100 thresholds in only 5.19 ms. The final version manages to compute contours at 100 thresholds in less than double the time it took the initial one to compute them at only one. We can also get more view of the final implementation's performance at actual runtime by measuring it with CUDA events. Averaged across 100 runs it takes the kernel 3.15 ms to finish its computations.

To get a more realistic feeling for the kernel's general behavior, we would need to also profile it on different instantiations of the height grid. With the one we used, we never saw how two-segment cases or multiple output contours per $2\times2$ subgrid affect the kernel. Ultimately, the performance will depend vastly on the height grid that the kernel is used on. We picked the one we deemed most realistic and estimate the mentioned cases to occur so rarely on similar height grids and threshold distributions, that it won't affect the conclusions we came to much. Still, a more general view would need to account for different scenarios as well.

#### Outlook
In the previous smoothstep kernel, we managed to get a good increase in performance by restricting the kernel to only run on the region affected by a user's mouse click. This is an optimization, we never considered for the marching squares kernel. Initially, we would need to run the kernel on the full height grid to obtain the contours for the whole of it. From there on, same as the smoothstep kernel only needs to consider the user's click region, the marching squares kernel would only need to consider the subgrids overlapping the height grid's modified region. The contours produced by all other subgrids are guaranteed to remain unchanged.

The difficulty to implement such an approach also for the marching squares kernel, lies in the dynamic output size of its output contours. Even if only one of the height grid's vertices changed, it could lead to a shift of the entire output array. We leave the experimentation with this idea for the future and now only give a very rough description. We expect such an approach to yield even further gains in performance.

In such implementation, unlike the current one, we would need to fix the output order of computed contours or at least implement a stable intermediate representation that is ordered with respect to the subgrids' positions in the grid. Such an implementation would closer resemble a two phased stable filter. The region lying before the modified one stays fixed and the one lying after it is shifted to accommodate for the new layout requirements that the modified region introduces. Actual contour computation would then only need to happen on the modified region to output the updated contours of the height grid.
