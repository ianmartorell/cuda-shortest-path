![http://cuda-shortest-path.googlecode.com/files/cuda.png](http://cuda-shortest-path.googlecode.com/files/cuda.png)

**此程序为NVIDIA CUDA 2011校园编程大赛作品，只可用于学习，如需用于其他目的，请联系作者。**

# 基于CUDA的并行最短路径算法 #

## 简介 ##

本作品提出了4种CUDA实现的并行最短路径算法。它们分别基于Dijkstra、Bellman-Ford、
Delta-Stepping、Sparse Matrix-Vector Bellman-Ford。作品中的CUDA算法与当前性能最优的Boost库中的各种最短路径算法进行了比较，在时间和空间上都有较强的竞争力，且在大数据上性能优势更明显。

作品截图：

CPU Shortest Path Algorithm:

![http://cuda-shortest-path.googlecode.com/files/cpu.png](http://cuda-shortest-path.googlecode.com/files/cpu.png)

GPU Shortest Path Algorithm:

![http://cuda-shortest-path.googlecode.com/files/gpu.png](http://cuda-shortest-path.googlecode.com/files/gpu.png)

Program UI:

![http://cuda-shortest-path.googlecode.com/files/ui.png](http://cuda-shortest-path.googlecode.com/files/ui.png)
