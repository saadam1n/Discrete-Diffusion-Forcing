<p align="center">
    <img src="docs/assets/img/d2f/logo_lr.jpg" width="300">
</p>

## Discrete Diffusion Forcing (D2F): dLLMs Can Do Faster-Than-AR Inference

<p align="center">
  <a href="Discrete Diffusion Forcing.pdf"><b>📄 Paper</b></a> •
  <a href="https://zhijie-group.github.io/Discrete-Diffusion-Forcing/"><b>📝 Blog Post</b></a> •
  <a href="https://huggingface.co/spaces/zhijie3/D2F-LLaDA-Instruct-8B"><b>🚀 Online Demo</b></a> •
  <a href="https://huggingface.co/SJTU-Deng-Lab/D2F_Dream_Base_7B_Lora"><b>🤗 D2F-Dream LoRA</b></a> •
  <a href="https://huggingface.co/SJTU-Deng-Lab/D2F_LLaDA_Instruct_8B_Lora"><b>🤗 D2F-LLaDA LoRA</b></a> 
</p>

<p align="center">
  <a href="https://discord.gg/aDWgxT6S2q"><b>💬 Discord</b></a> •
  <a href="docs/assets/img/d2f/wechat.png"><b>💬 Wechat</b></a>
</p>



https://github.com/user-attachments/assets/d9de6450-68d6-4caf-85c2-c7f384395c42


<p align="center">
  <br>
  <small><b>Real-time generation demo:</b> our D2F model (left) uses parallel block decoding, while the AR baseline (right) generates tokens sequentially. This visualizes the source of D2F's significant throughput advantage.</small>
</p>

<hr>

<p align="center">
    <img src="docs/assets/img/d2f/fig1_main_result.png" width="800">
    <br>
    <small><b>Inference throughput comparison:</b> D2F dLLMs surpass similarly-sized AR models in inference speed for the first time, achieving up to a <b>2.5x speedup</b> over LLaMA3 and a <b>>50x speedup</b> over vanilla dLLM baselines (Speed tests conducted on NVIDIA A100-PCIe-40GB GPUs).</small>
</p>

**Discrete Diffusion Forcing (D2F)** is a novel training and inference paradigm that, for the first time, enables open-source Diffusion Language Models (dLLMs) to surpass their autoregressive (AR) counterparts in inference speed. By introducing a highly efficient AR-diffusion hybrid model, D2F achieves:
- Up to a **2.5x speedup** over leading AR models like LLaMA3-8B.
- A staggering **50x acceleration** over vanilla dLLM baselines.
- Comparable generation quality on standard reasoning and coding benchmarks.

This repository provides the code to reproduce our evaluation results and run generation demos.

## 🔥 News!
* Aug 8, 2025: We've released the inference code and training pipeline of D2F!
## Contents
- [🤔 How It Works](#-how-it-works)
- [📊 Performance Highlights](#-performance-highlights)
- [🚀 Usage Guide](#-usage-guide)
- [🙏 Acknowledgements](#-acknowledgements)
- [©️ Citation](#️-citation)

## 🤔 How It Works

D2F overcomes the historical speed bottlenecks of dLLMs (KV Cache incompatibility and strict sequential dependencies) by restructuring the generation process.

**1. Hybrid Architecture:** D2F employs a **block-wise causal attention** mechanism. Attention *within* a block is bidirectional, preserving rich local context, while attention *between* blocks is causal. This simple but powerful change makes the model fully compatible with the standard KV Cache, drastically reducing redundant computations.

**2. Efficient Training via Asymmetric Distillation:** Instead of training from scratch, we distill a powerful, pre-trained bidirectional dLLM (teacher) into our cache-friendly D2F model (student). The student learns to match the teacher's output with only a limited, causal view of the context.

<p align="center">
    <img src="docs/assets/img/d2f/fig3_overview.png" width="800">
    <br>
    <small><b>Overview of Discrete Diffusion Forcing (D2F):</b> A D2F model (student) with a KV-cache-friendly block-wise causal attention mask is trained to mimic a powerful, pre-trained bidirectional dLLM (teacher), efficiently inheriting its capabilities.</small>
</p>

**3. High-Throughput Pipelined Decoding:** D2F is trained to predict future blocks based on *partially incomplete* prefixes. This enables a **pipelined parallel decoding** algorithm during inference, where multiple blocks are refined simultaneously in an asynchronous workflow, maximizing GPU utilization and throughput.

<p align="center">
    <img src="docs/assets/img/d2f/fig4_pipeline.png" width="800">
    <br>
    <small><b>Visualization of our pipelined parallel decoding:</b> New blocks are dynamically added and decoded in parallel with their predecessors, moving from a conservative "semi-activated" state to an aggressive "fully-activated" state. This creates a continuous, high-throughput generation flow.</small>
</p>

https://github.com/user-attachments/assets/41a0176b-e4ae-4f8b-95a6-daed7af2a027

<p align="center">
  <br>
  <small><b>A slow-motion demonstration of the parallel decoding process within a single block of D2F. Watch as multiple tokens within the block are refined simultaneously, showcasing the efficiency of our approach.</small>
</p>

## 📊 Performance Highlights

We applied D2F to two popular open-source dLLMs: **LLaDA-Instruct-8B** and **Dream-Base-7B**. The results demonstrate massive speedups over baselines and previous SOTA acceleration methods, without compromising on quality.

#### Performance on LLaDA-Instruct-8B
<center>

<strong>GSM8K-4-shot</strong>
<table style="width:100%; border-collapse: collapse; text-align: center;">
  <thead style="background-color:#f2f2f2;">
    <tr>
      <th style="padding: 8px; border: 1px solid #ddd;">Method</th>
      <th style="padding: 8px; border: 1px solid #ddd;">TPS ↑</th>
      <th style="padding: 8px; border: 1px solid #ddd;">Latency (s) ↓</th>
      <th style="padding: 8px; border: 1px solid #ddd;">Gen. Length</th>
      <th style="padding: 8px; border: 1px solid #ddd;">Score ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">LLaDA-Instruct</td>
      <td style="padding: 8px; border: 1px solid #ddd;">7.2 <font color="green">(1.0x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">32.3 <font color="green">(1.0x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">231</td>
      <td style="padding: 8px; border: 1px solid #ddd;">77.4</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">dLLM-Cache</td>
      <td style="padding: 8px; border: 1px solid #ddd;">20.1 <font color="green">(2.8x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">11.5 <font color="green">(2.8x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">231</td>
      <td style="padding: 8px; border: 1px solid #ddd;">77.5</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">Fast-dLLM (Prefix-Cache)</td>
      <td style="padding: 8px; border: 1px solid #ddd;">33.3 <font color="green">(4.6x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">7.0 <font color="green">(4.6x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">232</td>
      <td style="padding: 8px; border: 1px solid #ddd;">77.8</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">Fast-dLLM (Dual-Cache)</td>
      <td style="padding: 8px; border: 1px solid #ddd;">35.2 <font color="green">(4.9x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">6.6 <font color="green">(4.9x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">232</td>
      <td style="padding: 8px; border: 1px solid #ddd;"><b>78.9</b></td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;"><strong>D2F-LLaDA</strong></td>
      <td style="padding: 8px; border: 1px solid #ddd;"><strong>52.5 <font color="green">(7.3x)</font></strong></td>
      <td style="padding: 8px; border: 1px solid #ddd;"><strong>2.8 <font color="green">(11.5x)</font></strong></td>
      <td style="padding: 8px; border: 1px solid #ddd;">144</td>
      <td style="padding: 8px; border: 1px solid #ddd;">77.3</td>
    </tr>
  </tbody>
</table>

<strong>MBPP-3-shot</strong>
<table style="width:100%; border-collapse: collapse; text-align: center;">
  <thead style="background-color:#f2f2f2;">
    <tr>
      <th style="padding: 8px; border: 1px solid #ddd;">Method</th>
      <th style="padding: 8px; border: 1px solid #ddd;">TPS ↑</th>
      <th style="padding: 8px; border: 1px solid #ddd;">Latency (s) ↓</th>
      <th style="padding: 8px; border: 1px solid #ddd;">Gen. Length</th>
      <th style="padding: 8px; border: 1px solid #ddd;">Score ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">LLaDA-Instruct</td>
      <td style="padding: 8px; border: 1px solid #ddd;">0.9 <font color="green">(1.0x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">71.4 <font color="green">(1.0x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">65</td>
      <td style="padding: 8px; border: 1px solid #ddd;"><b>39.0</b></td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">dLLM-Cache</td>
      <td style="padding: 8px; border: 1px solid #ddd;">2.3 <font color="green">(2.6x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">28.3 <font color="green">(2.5x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">66</td>
      <td style="padding: 8px; border: 1px solid #ddd;">37.0</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">Fast-dLLM (Prefix-Cache)</td>
      <td style="padding: 8px; border: 1px solid #ddd;">13.0 <font color="green">(14.4x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">4.9 <font color="green">(14.6x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">64</td>
      <td style="padding: 8px; border: 1px solid #ddd;">37.6</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">Fast-dLLM (Dual-Cache)</td>
      <td style="padding: 8px; border: 1px solid #ddd;">15.3 <font color="green">(17.0x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">3.8 <font color="green">(18.8x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">58</td>
      <td style="padding: 8px; border: 1px solid #ddd;">36.4</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;"><strong>D2F-LLaDA</strong></td>
      <td style="padding: 8px; border: 1px solid #ddd;"><strong>47.6 <font color="green">(52.9x)</font></strong></td>
      <td style="padding: 8px; border: 1px solid #ddd;"><strong>1.4 <font color="green">(51.0x)</font></strong></td>
      <td style="padding: 8px; border: 1px solid #ddd;">68</td>
      <td style="padding: 8px; border: 1px solid #ddd;">38.0</td>
    </tr>
  </tbody>
</table>

<strong>HumanEval-0-shot</strong>
<table style="width:100%; border-collapse: collapse; text-align: center;">
  <thead style="background-color:#f2f2f2;">
    <tr>
      <th style="padding: 8px; border: 1px solid #ddd;">Method</th>
      <th style="padding: 8px; border: 1px solid #ddd;">TPS ↑</th>
      <th style="padding: 8px; border: 1px solid #ddd;">Latency (s) ↓</th>
      <th style="padding: 8px; border: 1px solid #ddd;">Gen. Length</th>
      <th style="padding: 8px; border: 1px solid #ddd;">Score ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">LLaDA-Instruct</td>
      <td style="padding: 8px; border: 1px solid #ddd;">2.8 <font color="green">(1.0x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">38.8 <font color="green">(1.0x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">107</td>
      <td style="padding: 8px; border: 1px solid #ddd;">36.0</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">dLLM-Cache</td>
      <td style="padding: 8px; border: 1px solid #ddd;">4.5 <font color="green">(1.6x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">23.3 <font color="green">(1.7x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">104</td>
      <td style="padding: 8px; border: 1px solid #ddd;">39.0</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">Fast-dLLM (Prefix-Cache)</td>
      <td style="padding: 8px; border: 1px solid #ddd;">13.7 <font color="green">(4.9x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">7.4 <font color="green">(5.2x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">102</td>
      <td style="padding: 8px; border: 1px solid #ddd;">38.4</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">Fast-dLLM (Dual-Cache)</td>
      <td style="padding: 8px; border: 1px solid #ddd;">19.2 <font color="green">(6.9x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">5.2 <font color="green">(7.5x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">100</td>
      <td style="padding: 8px; border: 1px solid #ddd;">35.4</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;"><strong>D2F-LLaDA</strong></td>
      <td style="padding: 8px; border: 1px solid #ddd;"><strong>81.6 <font color="green">(29.1x)</font></strong></td>
      <td style="padding: 8px; border: 1px solid #ddd;"><strong>1.6 <font color="green">(24.3x)</font></strong></td>
      <td style="padding: 8px; border: 1px solid #ddd;">133</td>
      <td style="padding: 8px; border: 1px solid #ddd;"><b>40.2</b></td>
    </tr>
  </tbody>
</table>

<strong>Math-4-shot</strong>
<table style="width:100%; border-collapse: collapse; text-align: center;">
  <thead style="background-color:#f2f2f2;">
    <tr>
      <th style="padding: 8px; border: 1px solid #ddd;">Method</th>
      <th style="padding: 8px; border: 1px solid #ddd;">TPS ↑</th>
      <th style="padding: 8px; border: 1px solid #ddd;">Latency (s) ↓</th>
      <th style="padding: 8px; border: 1px solid #ddd;">Gen. Length</th>
      <th style="padding: 8px; border: 1px solid #ddd;">Score ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">LLaDA-Instruct</td>
      <td style="padding: 8px; border: 1px solid #ddd;">21.1 <font color="green">(1.0x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">11.5 <font color="green">(1.0x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">243</td>
      <td style="padding: 8px; border: 1px solid #ddd;">23.7</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">dLLM-Cache</td>
      <td style="padding: 8px; border: 1px solid #ddd;">26.9 <font color="green">(1.3x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">9.1 <font color="green">(1.3x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">246</td>
      <td style="padding: 8px; border: 1px solid #ddd;">23.2</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">Fast-dLLM (Prefix-Cache)</td>
      <td style="padding: 8px; border: 1px solid #ddd;">47.7 <font color="green">(2.3x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">5.2 <font color="green">(2.2x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">246</td>
      <td style="padding: 8px; border: 1px solid #ddd;">22.4</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">Fast-dLLM (Dual-Cache)</td>
      <td style="padding: 8px; border: 1px solid #ddd;">42.5 <font color="green">(2.0x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">5.8 <font color="green">(2.0x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">246</td>
      <td style="padding: 8px; border: 1px solid #ddd;">22.4</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;"><strong>D2F-LLaDA</strong></td>
      <td style="padding: 8px; border: 1px solid #ddd;"><strong>90.2 <font color="green">(4.3x)</font></strong></td>
      <td style="padding: 8px; border: 1px solid #ddd;"><strong>4.3 <font color="green">(2.7x)</font></strong></td>
      <td style="padding: 8px; border: 1px solid #ddd;">384</td>
      <td style="padding: 8px; border: 1px solid #ddd;"><b>29.1</b></td>
    </tr>
  </tbody>
</table>
<br>
<small>D2F provides transformative speedups for LLaDA-Instruct-8B, achieving a <b>52.9x</b> increase in throughput on MBPP and a <b>29.1x</b> increase on HumanEval while also improving the score.</small>

</center>

#### Performance on Dream-Base-7B
<center>

<strong>GSM8K-CoT-8-shot</strong>
<table style="width:100%; border-collapse: collapse; text-align: center;">
  <thead style="background-color:#f2f2f2;">
    <tr>
      <th style="padding: 8px; border: 1px solid #ddd;">Method</th>
      <th style="padding: 8px; border: 1px solid #ddd;">TPS ↑</th>
      <th style="padding: 8px; border: 1px solid #ddd;">Latency (s) ↓</th>
      <th style="padding: 8px; border: 1px solid #ddd;">Gen. Length</th>
      <th style="padding: 8px; border: 1px solid #ddd;">Score ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">Dream-Base</td>
      <td style="padding: 8px; border: 1px solid #ddd;">9.5 <font color="green">(1.0x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">26.8 <font color="green">(1.0x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">255</td>
      <td style="padding: 8px; border: 1px solid #ddd;">75.0</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">dLLM-Cache</td>
      <td style="padding: 8px; border: 1px solid #ddd;">26.0 <font color="green">(2.7x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">9.8 <font color="green">(2.7x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">255</td>
      <td style="padding: 8px; border: 1px solid #ddd;">72.0</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">Fast-dLLM (Prefix-Cache)</td>
      <td style="padding: 8px; border: 1px solid #ddd;">50.3 <font color="green">(5.3x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">5.1 <font color="green">(5.3x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">255</td>
      <td style="padding: 8px; border: 1px solid #ddd;">76.6</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">Fast-dLLM (Dual-Cache)</td>
      <td style="padding: 8px; border: 1px solid #ddd;">49.8 <font color="green">(5.2x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">5.1 <font color="green">(5.3x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">255</td>
      <td style="padding: 8px; border: 1px solid #ddd;">75.0</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;"><strong>D2F-Dream</strong></td>
      <td style="padding: 8px; border: 1px solid #ddd;"><strong>91.2 <font color="green">(9.6x)</font></strong></td>
      <td style="padding: 8px; border: 1px solid #ddd;"><strong>2.8 <font color="green">(9.6x)</font></strong></td>
      <td style="padding: 8px; border: 1px solid #ddd;">256</td>
      <td style="padding: 8px; border: 1px solid #ddd;"><b>77.6</b></td>
    </tr>
  </tbody>
</table>

<strong>MBPP-3-shot</strong>
<table style="width:100%; border-collapse: collapse; text-align: center;">
  <thead style="background-color:#f2f2f2;">
    <tr>
      <th style="padding: 8px; border: 1px solid #ddd;">Method</th>
      <th style="padding: 8px; border: 1px solid #ddd;">TPS ↑</th>
      <th style="padding: 8px; border: 1px solid #ddd;">Latency (s) ↓</th>
      <th style="padding: 8px; border: 1px solid #ddd;">Gen. Length</th>
      <th style="padding: 8px; border: 1px solid #ddd;">Score ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">Dream-Base</td>
      <td style="padding: 8px; border: 1px solid #ddd;">10.4 <font color="green">(1.0x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">24.6 <font color="green">(1.0x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">256</td>
      <td style="padding: 8px; border: 1px solid #ddd;">56.2</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">dLLM-Cache</td>
      <td style="padding: 8px; border: 1px solid #ddd;">25.5 <font color="green">(2.5x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">10.0 <font color="green">(2.5x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">256</td>
      <td style="padding: 8px; border: 1px solid #ddd;">52.6</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">Fast-dLLM (Prefix-Cache)</td>
      <td style="padding: 8px; border: 1px solid #ddd;">71.6 <font color="green">(6.9x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">3.6 <font color="green">(6.8x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">256</td>
      <td style="padding: 8px; border: 1px solid #ddd;"><b>56.4</b></td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">Fast-dLLM (Dual-Cache)</td>
      <td style="padding: 8px; border: 1px solid #ddd;">73.2 <font color="green">(7.0x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">3.5 <font color="green">(7.0x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">256</td>
      <td style="padding: 8px; border: 1px solid #ddd;">51.0</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;"><strong>D2F-Dream</strong></td>
      <td style="padding: 8px; border: 1px solid #ddd;"><strong>105 <font color="green">(10.1x)</font></strong></td>
      <td style="padding: 8px; border: 1px solid #ddd;"><strong>2.3 <font color="green">(10.7x)</font></strong></td>
      <td style="padding: 8px; border: 1px solid #ddd;">240</td>
      <td style="padding: 8px; border: 1px solid #ddd;">55.2</td>
    </tr>
  </tbody>
</table>

<strong>HumanEval-0-shot</strong>
<table style="width:100%; border-collapse: collapse; text-align: center;">
  <thead style="background-color:#f2f2f2;">
    <tr>
      <th style="padding: 8px; border: 1px solid #ddd;">Method</th>
      <th style="padding: 8px; border: 1px solid #ddd;">TPS ↑</th>
      <th style="padding: 8px; border: 1px solid #ddd;">Latency (s) ↓</th>
      <th style="padding: 8px; border: 1px solid #ddd;">Gen. Length</th>
      <th style="padding: 8px; border: 1px solid #ddd;">Score ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">Dream-Base</td>
      <td style="padding: 8px; border: 1px solid #ddd;">20.2 <font color="green">(1.0x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">12.6 <font color="green">(1.0x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">255</td>
      <td style="padding: 8px; border: 1px solid #ddd;">54.3</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">dLLM-Cache</td>
      <td style="padding: 8px; border: 1px solid #ddd;">23.2 <font color="green">(1.1x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">11.0 <font color="green">(1.1x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">255</td>
      <td style="padding: 8px; border: 1px solid #ddd;"><b>55.5</b></td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">Fast-dLLM (Prefix-Cache)</td>
      <td style="padding: 8px; border: 1px solid #ddd;">62.4 <font color="green">(3.1x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">4.1 <font color="green">(3.1x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">255</td>
      <td style="padding: 8px; border: 1px solid #ddd;">54.3</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">Fast-dLLM (Dual-Cache)</td>
      <td style="padding: 8px; border: 1px solid #ddd;">60.0 <font color="green">(3.0x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">4.3 <font color="green">(2.9x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">255</td>
      <td style="padding: 8px; border: 1px solid #ddd;">53.0</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;"><strong>D2F-Dream</strong></td>
      <td style="padding: 8px; border: 1px solid #ddd;"><strong>73.2 <font color="green">(3.6x)</font></strong></td>
      <td style="padding: 8px; border: 1px solid #ddd;"><strong>3.1 <font color="green">(4.1x)</font></strong></td>
      <td style="padding: 8px; border: 1px solid #ddd;">227</td>
      <td style="padding: 8px; border: 1px solid #ddd;">54.3</td>
    </tr>
  </tbody>
</table>

<strong>Math-4-shot</strong>
<table style="width:100%; border-collapse: collapse; text-align: center;">
  <thead style="background-color:#f2f2f2;">
    <tr>
      <th style="padding: 8px; border: 1px solid #ddd;">Method</th>
      <th style="padding: 8px; border: 1px solid #ddd;">TPS ↑</th>
      <th style="padding: 8px; border: 1px solid #ddd;">Latency (s) ↓</th>
      <th style="padding: 8px; border: 1px solid #ddd;">Gen. Length</th>
      <th style="padding: 8px; border: 1px solid #ddd;">Score ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">Dream-Base</td>
      <td style="padding: 8px; border: 1px solid #ddd;">9.9 <font color="green">(1.0x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">25.8 <font color="green">(1.0x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">256</td>
      <td style="padding: 8px; border: 1px solid #ddd;">35.8</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">dLLM-Cache</td>
      <td style="padding: 8px; border: 1px solid #ddd;">12.7 <font color="green">(1.3x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">20.2 <font color="green">(1.3x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">256</td>
      <td style="padding: 8px; border: 1px solid #ddd;">34.5</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">Fast-dLLM (Prefix-Cache)</td>
      <td style="padding: 8px; border: 1px solid #ddd;">65.6 <font color="green">(6.6x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">3.9 <font color="green">(6.6x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">256</td>
      <td style="padding: 8px; border: 1px solid #ddd;"><b>37.6</b></td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;">Fast-dLLM (Dual-Cache)</td>
      <td style="padding: 8px; border: 1px solid #ddd;">67.0 <font color="green">(6.8x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">3.8 <font color="green">(6.8x)</font></td>
      <td style="padding: 8px; border: 1px solid #ddd;">256</td>
      <td style="padding: 8px; border: 1px solid #ddd;">37.1</td>
    </tr>
    <tr>
      <td style="padding: 8px; border: 1px solid #ddd;"><strong>D2F-Dream</strong></td>
      <td style="padding: 8px; border: 1px solid #ddd;"><strong>98.8 <font color="green">(10.0x)</font></strong></td>
      <td style="padding: 8px; border: 1px solid #ddd;"><strong>2.6 <font color="green">(9.9x)</font></strong></td>
      <td style="padding: 8px; border: 1px solid #ddd;">256</td>
      <td style="padding: 8px; border: 1px solid #ddd;">35.4</td>
    </tr>
  </tbody>
</table>
<br>
<small>Applying D2F to Dream-Base-7B results in substantial gains, including a <b>9.6x</b> speedup on GSM8K-CoT and a <b>10.1x</b> speedup on MBPP. Notably, performance scores often improve alongside the acceleration.</small>
</center>

## 🚀 Usage Guide

### 1. Installation

First, clone the repository and set up the environment.

```bash
# Clone the repository
git clone https://github.com/zhijie-group/Discrete-Diffusion-Forcing.git
cd Discrete-Diffusion-Forcing

# Create and activate a conda environment
conda create -n d2f python=3.10
conda activate d2f

# Install dependencies
pip install -r requirements.txt
```

### 2. Evaluation
All evaluation scripts are located in the `D2F-eval` directory.

```bash
cd D2F-eval
```
To evaluate the **D2F-Dream** model on all benchmarks, run:
```bash
bash eval_dream.sh
```

To evaluate the **D2F-LLaDA** model on all benchmarks, run:
```bash
bash eval_llada.sh
```
The results will be saved in the `output_path` specified within the shell scripts.

> ### ❗️ Important Notice for HumanEval
> The `HumanEval` benchmark requires a post-processing step to sanitize the generated code and calculate the final `pass@1` score. After the evaluation script finishes, run the following command:
> ```bash
> python postprocess_code.py {path/to/your/samples_humaneval_xxx.jsonl}
> ```
> Replace the path with the actual path to your generated samples file, which can be found in the specified `output_path`.

### 3. Generation Demo

We provide simple scripts to demonstrate the generation process and compare D2F with a standard AR baseline.
```bash
# To run a demo with the baseline AR generation method:
python generate_llada_demo_ar.py

# To run a demo with the D2F pipelined block generation method:
python generate_llada_demo_block.py
```
You can inspect these files to see how to use the D2F model for inference in your own projects.

## 📚 Future Works

- [ ] Implement dLLM-suported vLLM

- [ ] Implement dLLM specific decoding kernel with kv cache loading

...

## 🙏 Acknowledgements
Our work builds upon the foundations laid by the original **LLaDA** and **Dream** models. We thank their authors for making their work public. We are also grateful for the powerful open-source tools from Hugging Face that made this research possible.

## ©️ Citation
If you find our work useful for your research, please consider citing our paper:
```bibtex
@misc{wang2025d2f,
  title        = {Diffusion LLMs Can Do Faster-Than-AR Inference via Discrete Diffusion Forcing},
  author       = {Wang, Xu and Xu, Chenkai and Jin, Yijie and Jin, Jiachun and Hu, Yanzhe and Deng, Zhijie},
  year         = {2024},
  howpublished = {\url{https://github.com/zhijie-group/Discrete-Diffusion-Forcing/blob/main/Discrete%20Diffusion%20Forcing.pdf}},
  note         = {Accessed: 2025-08-13}
}

```
