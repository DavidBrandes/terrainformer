## GPU Speed of Light Throughput
### Compute (SM) Throughput
How close are the SMs to their peak performance and not idling. In what percentage of cycles are the SMs actually doing work.

### Memory / L1/TEX Cache / L2 Cache / DRAM Throughput
To what percentage are the memory units operating to their peak capacity. A lower value means that more data could be served while a higher value indicated that the kernel already is reaching the limits.

### Duration
The kernels execution time

### Roofline Graphs
The x axis shows the arithmetic intensity (FLOP per byte fetched) and the y axis the hardware perfomance (FLOP per second). Since the GPU has a limit on both the memory bandwith and compute intensiity, the lines show the theoretical limits of the GPU. The arithmetic intesity is given for the different memory units (L1, L2, DRAM).

## Compute Workload Analysis
### Executed IPC Elapsed 
How many instructions where retired per clock cycle on average over all cycles

### SM Busy / Issue Slots Busy
The fraction of cycles during which the warp scheduler issued at least one instruction as opposed to beinf stalled. These two metrics are seem from different points (SM level vs. warp scheduler level) but mean essentially the same thing.

## Memory Workload Analysis
### Memory Throughput
The amount of data transfered for the specified time unit

### L1/TEX / L2 Hit Rate
What percentag of requests reaching this unit were hits or had to be forwarded to subsequent ones

### Mem Busy
How often did the memory piplines actually have one active request

### Max Bandwith
This measures the amount of data flowing thoruhg the pipelines as compared to the peak capacity

### Mem pipes busy
This measures to what percentage the pipelines are busy when seen from the SMs perspective, i.e. how busy are the load/store units

### Load / Store Efficiency
`smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio ` and `smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio`. Out of every byte fetched from a sector in global memory, how many were actually used. Out of every store instructio, how many bytes were actually written.

## Scheduler Statistics
### Active Warps Per Scheduler
How many warps are residing on a scheduler on average. These not necessarily need to be warps doing anything useful.

### Eligible Warps Per Scheduler
Out of all the active warps on a scheduler, how many are actaully eligible to do some work on average

### Issued Warps Per Scheduler
How many of the warps were actually issued an instruction per cycle (max 1 per scheduler per cycle).

### No Eligible / One or More Eligible
In each cycle, with what percentage would each SM have no / at least one eligible warp to do work with

### Notes
The fundamental aim of GPU optimization is to maximize the issued warps per schduler. This is often achieved by latency hiding, i.e. by keeping the number of eligible warps per cycle high so that there is always some work to do.

If this number of issued warps is lower than the elibible warps per schduler, it could indicate that warps became active in bursts and there are periods in between where the scheduler had no warps to do work with.

## Warp State Statistics
### Warp Stall
With every issued warp instruction, how many cycles were spent stalling due to the specified reason
#### Stall LG Throttle
Stands for Local  Global throttle and happens when the memory pipeline of a SM is already overloaded and such requests are therefore queued
#### Stall Long Scoreboard
When a memory instruction was issued, how many cycles were spent stalling before the warp was able to continue
#### Stall MIO Throttle
Stands for Memory Input/Output and sits between the SM load/store pipe and the rest of the memory hierachry (L1, L2, etc..). Indicates that the units internal queue was full.
