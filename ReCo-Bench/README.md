---
license: cc-by-nc-sa-4.0
task_categories:
- image-to-video
tags:
- video-editing
- instruction-following
- evaluation-benchmark
---

# ReCo-Bench

[**Project Page**](https://zhw-zhang.github.io/ReCo-page/) | [**Paper**](https://huggingface.co/papers/2512.17650) | [**GitHub**](https://github.com/HiDream-ai/ReCo) | [**ReCo-Data**](https://huggingface.co/datasets/HiDream-ai/ReCo-Data)

This is the official **ReCo-Bench** dataset introduced in the paper "[Region-Constraint In-Context Generation for Instructional Video Editing](https://huggingface.co/papers/2512.17650)". ReCo-Bench is a VLLM-based evaluation benchmark designed to comprehensively and effectively assess video editing quality.

## Usage

After downloading the repository, you can start the evaluation directly by running the following script:

```bash
bash run_eval_via_gemini.sh
```

# VLLM-based Evaluation Benchmark

Traditional video generation metrics often struggle to accurately assess the fidelity and quality of video editing. Inspired by recent image editing evaluation protocols, we propose a VLLM-based evaluation benchmark to comprehensively and effectively assess video editing quality.

## Testing Data

We collect 480 video-instruction pairs as the testing data, distributed evenly with 120 pairs for each of the four tasks (i.e., object add, remove, replace, and video stylization). All source videos are collected from the Pexels video platform. For local editing tasks (i.e., object add, remove, and replace), we utilize Gemini-2.5-Flash-Thinking to brainstorm and generate diverse editing instructions based on the video content. For rigorous evaluation on video stylization, we randomly select 10 source videos and apply 12 distinct styles to each, resulting in 120 evaluation pairs.

## Evaluation Metrics

While previous image-based metrics primarily focus on editing accuracy and static generation quality, evaluating video editing entails greater complexity. To address this, we construct a diverse set of evaluation dimensions specifically tailored for video. The corresponding system prompt designed for the VLLM evaluates performance across three major perspectives, comprising a total of nine sub-dimensions:

- **Edit Accuracy** (S_EA): Evaluate how well the result aligns with the instruction.
  - **Semantic Accuracy (SA):** Does the edited video correctly follow the semantics of the text instruction?
  - **Scope Precision (SP):** Is the editing confined strictly to the target region without affecting the background?
  - **Content Preservation (CP):** Are the non-edited regions or original details faithfully preserved? (For stylization, this corresponds to structural preservation.)

- **Video Naturalness** (S_VN): Evaluates the realism and coherence of the generated content.
  - **Appearance Naturalness (AN):** Are the lighting, texture, and color of the edited video natural?
  - **Scale Naturalness (SN):** Is the size and proportion of the edited object reasonable relative to the environment? (For stylization, this captures cases where the stylized object becomes unreasonably large.)
  - **Motion Naturalness (MN):** Does the movement of the edited object (or the style rendering) follow physically plausible dynamics?

- **Video Quality** (S_VQ): Evaluates the fundamental visual quality of the edited video.
  - **Visual Fidelity (VF):** Is the video clear, sharp, and free from visual artifacts?
  - **Temporal Stability (TS):** Is the video free from flickering or jittering across frames?
  - **Edit Stability (ES):** Is the edited content consistently preserved in identity and appearance throughout the video duration?

The VLLM rates the score for each sub-dimension from 0 to 10. Then, we attain the per-category scores (i.e., S_EA, S_VN, S_VQ) by calculating the geometric mean of their respective sub-dimensions as:

$$
\begin{aligned}
S_{EA} &= \sqrt[3]{SA \cdot SP \cdot CP}, \\
S_{VN} &= \sqrt[3]{AN \cdot SN \cdot MN}, \\
S_{VQ} &= \sqrt[3]{VF \cdot TS \cdot ES}.
\end{aligned}
$$

Finally, the overall score S is calculated as the arithmetic mean of the three per-category scores:

$$
S = \frac{1}{3}\left(S_{EA} + S_{VN} + S_{VQ}\right).
$$

## Citation

If you find ReCo-Bench useful for your research, please cite:

```bibtex
@article{reco,
	title={{Region-Constraint In-Context Generation for Instructional Video Editing}},
	author={Zhongwei Zhang and Fuchen Long and Wei Li and Zhaofan Qiu and Wu Liu and Ting Yao and Tao Mei},
	journal={arXiv preprint arXiv:2512.17650},
	year={2025}
}
```