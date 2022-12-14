### Classification in terms of data and instructions

|                       | Same Data | Multiple Data |
| --------------------- | --------- | ------------- |
| Single Instruction    | SISD (Serial Code)     | SIMD          |
| Multiple Instructions | MISD      | MIMD          | 

1. SISD
	1. Compute $F_1 = \sum_n f_n$
	2. Algorithm: 
		1. $F_1 = f_1$
		2. $F_1 += f_2$
		4. ...

2. SIMD
	1. Compute $\vec{A} = 4 \vec{B}$
	2. Algorithm:
| CPU1          | CPU2          | CPU3          |
| ------------- | ------------- | ------------- |
| A[1] = 4 B[1] | A[2] = 4 B[2] | A[3] = 4 B[3] |
| A[4] = 4 B[4] | A[5] = 4 B[5] | A[6] = 4 B[6] |
| ...           | ...           | ...           | 

3. MISD
	1. Compute: $F_\mu = \frac{1}{N} \sum_n^N f_n^\mu$ (Moments problem)
| CPU1         | CPU2           | CPU3           | ... |
| ------------ | -------------- | -------------- | --- |
| $F_1 = f_1$  | $F_2 = f_1^2$  | $F_3 = f_1^3$  | ... |
| $F_1 += f_2$ | $F_2 += f_2^2$ | $F_3 += f_2^3$ | ... |
| ...          | ...            | ...            | ... | 

4. MIMD
	1. Task: Different CPUS have different tasks and act on different data sets
	2. Ex: CPU1 - CPU10 is computing and saving data, CPU11 - CPU20 is performing calculations

### Classification in terms of memory

1. Shared memory (typical of modern laptops / desktops)
	1. All processor cores access the same pool of memory
2. Distributed Memory (SHARCNET cluster)
	1. Each processor has it's own block of memory that is only accessible by that processor

## MPI (Message Passing Interface)
Distributed memory model

Process:
1. Initialize
2. Find # of processes
3. Determine which process I am
4. Send a message
5. Receive a message
6. Terminate MPI

