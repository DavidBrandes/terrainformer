## RTX 2000 Ada Generation Laptop GPU

Peak throughput: 14.5 TFLOPS
Peak memory bandwidth: 256 GB/s
Measured memory bandwidth: 222 GB/s
Ridge Point: 56.6 FLOP/B
Grid Size: 8000x4000
SMs: 24
Max threads per SM: 1536
L2 size: 33554432 bytes
## Kernel performance optimizations
### Smoothstep kernel
To get warmed up, we start our optimization journey off with the relatively simple [smoothstep](../src/compute/kernels/smoothstep.cu) kernel. This element wise kernel is used by the application on a circular region of radius $r$ around the users click position $c$ on the height grid. It modifies a grid's vertex $v$ with height $h_v$ and distance $d_v=\Vert{v - c}\Vert_2$ to the click position $c$ by an amount $m_v$, which we compute as
$$
m_v = 3\alpha f_v^2 - 2\alpha f_v^3 ,\quad\text{where }
f_v = \begin{cases} 1 - \frac{d_v}{r}, & d_v < r \\ 0, & d_v \geq r\end{cases}\text{ and }\alpha \in \mathbb{R}.
$$
The value $\alpha$ is some scaling factor which controls by how much and in which direction the height should be modified. 

Each click then iteratively updates the height grid's vertices as
$$
h_v^i=h_v^{i-1}+m_v^i,
$$
where $h_v^i$ and $m_v^i$ denote the height and modification of vertex $v$ after the $i$-th click, respectively. Notice how vertices closer to the click position have a bigger modification applied to them than vertices further away. Also, vertices outside the clicks range $r$ aren't modified at all as their modification evaluates to $0$.

To benchmark this kernel, we simulate a click centered on our grid of $8000\times4000$ vertices with a radius of half the grids height. We start profiling with a naive kernel implementation that computes one vertex per thread and uses a blocks of size $16\times16$. From this starting point, we will work our way towards more performant variations of this kernel.

We can already notice, that in this naive implementation, most threads won't be doing any work at all. Most of the grid's vertices will lie outside the click's effect radius and hence will evaluate to a modification $m_v$ of $0$. In particular we can estimate that only a fraction of $\pi r^2/8r^2\approx 0.39$ of all threads will be doing useful work.

Let us also analyse the kernels theoretical performance. For a vertex that has it's height modified, we need to perform 16 floating point operations. This stands against one load and one store operation involving 8 bytes in total. Consequently, we get a computational intensity of $2\frac{\text{FLOP}}{\text{B}}$.

Even though the actually achievable computational throughput is way lower than the reported 14.5 TFLOPS (e.g. due to thermal throttling; ncu reports a ceiling of ~5.7 TFLOPS), we can still see that this kernel is clearly memory bound. If we take just the measured 5.7 TFLOPS, a perfect kernel would use only ~9% of the GPU's compute capacity.

Now we are ready for some optimizations. A simple, but important one, is to not use `powf` to compute squares and cubes and instead write out their product by hand. The built in power function is computationally quite heavy. Avoiding it brings us down to 40.655.248 from 218.676.401 issued instructions.

The next obvious optimization is to only avoid most of the unnecessary computations on vertices outside of the click's effect radius $r$. We can launch the kernel only for a sub-square of the whole grid which encloses the click's circle. This modification brings down the kernels overall run time to 571 µs from 785 µs. With this new square of side length $2r$, we also increase the fraction of threads doing useful work to $\pi r^2/4r^2\approx 0.79$.

Since there are no cross dependencies for the vertices in our grid and we are mostly memory bound, we cannot do much except optimizing that access to memory. With our existing block configuration of $16\times16$ threads, the heights are already loaded and stored in a fairly coalesced way.
though we could not attribute it to a specific metric.

We observe a slight improvement in performance using a $32×8$ block configuration. Though we could not attribute it to a specific metric. With a last tweak we transpose this configuration to $8\times32$ and add in vector loads along the rows. This gives us a nice square access of $32\times32$ elements per block. For specific brush radii, this should give us the minmal amount of blocks required to cover the entire clicks circle.

This last configuration leaves us at a runtime of 541 µs. We observe a increased throughput of between 29% and 70% in L1, L2 and DRAM when compared to a unvectorized implementation. Memory is now busier and we utilize its bandwidth at 92%, which is as good as we could manage with this memory bound kernel. With this vectorized kernel we now also get more active warps on average. 

Using `float4` loads and stores, we in a sense coarsened our kernel, giving each thread more work to do and keeping it busy while others might wait on their data to arrive. All of that while issuing fewer memory instructions than with an unvectorized implementation. This last modification gets us pretty close to what can actually be achieved with our GPU. Using the stated memory bandwidth of 256 GB/s, an optimal kernel would take 500 µs for our grid of $4000\times4000$ elements (considering both loads and stores).

In theory we could also try to avoid the square root in the computation of the distance of a grid vertex to the click center. However simply leaving it out would give us only very small modifications at the circle boundaries. And even if we were ok with that, it would not give us much with our memory bound kernel. Having fewer steps to compute would simply leave the processors run idle more often.

When using this kernel during an actual application run, we can expect it to be launched multiple times in a row. On our GPU with a L2 cache size of 33,554,432 bytes, we can fit a total amount of 8,388,608 floating point values. In an optimal scenario, where our cache is not disturbed by anything else and we launch the kernel consecutively with our constrained block of $4000\times4000$ elements, around 52% of all data could theoretically be served hot from cache instead of DRAM.

To verify this behaviour, we benchmarked 100 identical runs of this kernel, once with flushing the cache before each run and once without. We could not observe any meaningful difference in runtime. This suggests that the kernel's access pattern flushes cache lines by itself before it could be reused by a subsequent run. Indeed, if we run the kernel with a smaller brush radius that fits all data into L2, we see the overall runtime decreased by 28%.
### Marching Squares kernel
We next take a look at the [marching squares](../src/compute/kernels/marching_squares.cu) kernel. This kernel uses the [marching squares algorithm](https://en.wikipedia.org/wiki/Marching_squares) to compute the contour lines of the underlying height grid. For the specified threshold values, this algorithm approximates respective contours with linear segments at a resolution of the grid underlying dimensions.

While it's basic idea can be compared to a convolution kernel, its difficulty lies in the dynamic size of the computed output. For every four bordering grid verticies (a $2\times2$ subgrid) and a single threshold, it may produce zero, one or two segments. This variation naturally increases with the amount of thresholds one wants to compute the contours at.

A simple solution to deal with this dynamic output would be to simply have each $2\times2$ subgrid always produce its maximal amount of output segments and set the unused ones to `NaN` or some values that places them somewhere outside the displayed area. However, that would take away a lot of the challenges for optimizing the kernel and what we are actually interested in. Plus, a kernel that yields only as many output segments as actually required gives a nice general solution and further takes away work the shader that renders them.

As before, we start with a naive solution and will work our way towards a more performant version. That is, we start of with the simplest version for the kernel: A block size of $16\times16$, using no shared memory and writing the segments to an output buffer indexed via an atomic counter. For the underlying height grid we again choose a size of $8000\times4000$. We also compute the contours for each threshold level sequentially one after another.

Let us take a look at the computational intensity of this kernel. Since the amount of computational steps, bytes loaded and stored differs with the amount of output segments, we present the different cases in the below table.

|                  | 0 Segments | 1 Segment | 2 Segments |
| ---------------- | ---------- | --------- | ---------- |
| **Bytes Loaded** | 16         | 20        | 20         |
| **Bytes Stored** | 0          | 20        | 36         |
| **FLOPs**        | 0          | 8         | 38         |
| **OP/B**         | 0          | 0.2       | 0.68       |
We can already notice that in this naive implementation, the kernel is heavily memory bound, even more so than the previous smoothstep kernel. Further we see that the computational intensity differs by quite a bit across the various output conditions.

Next, we would like to get a feeling about the frequency, in which we can expect each of these 3 cases to arise. We launch the kernel on very regular sinusoidal grid and two variations of Perlin noise (one producing very turbulent and the other more gentle terrain) and the the contour thresholds vary in between the range of possible values. When counting the occurrence of each case, we notice a very skewed distribution which we show in the below table.

|                            | 0 Segments | 1 Segment | 2 Segments |
| -------------------------- | ---------- | --------- | ---------- |
| **Sinusoidal**             | ~99.9%     | ~0.1%     | ~0%        |
| **Gentle Perlin Noise**    | ~99.8%     | ~0.2%     | ~0%        |
| **Turbulent Perlin Noise** | ~99.1%     | ~0.89%    | ~0.01%     |
The output is quite revealing. For most $2\times2$ subgrids, we won't be doing any computation at all. Only rarely, a subgrid will actually produce an output segment and two will be produced even rarer. As a gird initialized with gentle Perlin noise appears to be the more interesting and realistic case, we will use it to continue with our profiling.

#### Shared memory

With all of this theory in mind, let us now start our optimization journey. An obvious first starting point could be to use shared memory instead of loading all the data from DRAM for each thread. With  computation happening in  $2\times2$ subgrids, most data is actually reused by other threads. Also the number of halo cells is rather small.

Trying it in practice, we however observe a performance that is way worse than that of a naive implementation loading all data separately. While memory throughput is down across all layers, the use of shared memory now adds additional instructions (due to the branching logic when loading data) and more importantly, barriers to our kernel. We now observe way more warps stalling and doing nothing while they wait for their block's data to arrive. This is especially hurtful with the large number of threads that later end up doing no useful work anyways.


// 2D -> 1D; thread, warp, block scan -> atomic write (both on full and slim kernel)
// TODO try __restrict__, __ldg()
// Warp shuffle atomic add
// 0: 0.998288, 1: 0.001712, 2: 0.000000 across 21 thresholds
// Threshold 0.000000, 0: 31853485, 1: 134516, 2: 0
// 0: 0.995795, 1: 0.004205, 2: 0.000000


//
Compute capability → what the GPU can do (hardware feature set)
Driver version → how software communicates and runs on that GPU
CUDA toolkit → how code is compiled for that ecosystem
