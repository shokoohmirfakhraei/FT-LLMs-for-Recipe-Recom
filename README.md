Project 23: Fine-Tuning LLMs for Recipe Recommendation and Instruction Following
Authors:
Shokooh Mirfakhraei – University of Oulu, Finland
Seyedehsahar Fatemi Abhari – University of Oulu, Finland

Overview
This project fine-tunes a Large Language Model (TinyLlama-1.1B-Chat-v1.0) on the HUMMUS recipe dataset to improve recipe recommendation and instruction-following accuracy. It also integrates Retrieval-Augmented Generation (RAG) to reduce hallucinations and ground responses in real recipe data.

Methodology
Dataset: HUMMUS (507k+ recipes with nutrition info)
Fine-tuning: LoRA (parameter-efficient adaptation)
Framework: Unsloth (optimized for memory efficiency)

Fine-tuning improved fluency and accuracy, while RAG reduced hallucinations but slightly lowered fluency.
