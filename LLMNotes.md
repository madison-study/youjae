# 📚 LLM Notes

[(This notes are based on "LLM Course" by Hugging Face)](https://huggingface.co/learn/llm-course/en)

## 📑 Table of Contents

- [Natural Language Processing and Large Language Models](#natural-language-processing-and-large-language-models)
  - [What is NLP?](#what-is-nlp)
  - [Large Language Models (LLMs)](#large-language-models-llms)
  - [Limitations of LLMs](#limitations-of-llms)
  - [Challenges in Language Processing](#challenges-in-language-processing)
- [Transformers, what can they do?](#transformers-what-can-they-do)
  - [Capabilities](#capabilities)
  - [Pipeline Function](#pipeline-function)
  - [Modalities](#modalities)
- [How do Transformers work?](#how-do-transformers-work)
  - [History](#history)
  - [Transformer Architecture](#transformer-architecture)
  - [Attention Mechanism](#attention-mechanism)
  - [Transfer Learning](#transfer-learning)

# Natural Language Processing and Large Language Models

## What is NLP?

NLP is a field combining linguistics and machine learning to understand human language in context. Common NLP tasks include:

- Sentence classification (sentiment analysis, spam detection)
- Word-level classification (part-of-speech tagging, named entity recognition)
- Text generation
- Question answering
- Translation and summarization

## Large Language Models (LLMs)

LLMs have revolutionized NLP with models like GPT and Llama. Key characteristics:

- Massive scale (billions of parameters)
- General capabilities across multiple tasks
- In-context learning from examples in prompts
- Emergent abilities not explicitly programmed

## Limitations of LLMs

- Hallucinations (generating incorrect information)
- Lack of true understanding
- Bias from training data
- Limited context windows
- High computational requirements

## Challenges in Language Processing

Text processing for machines differs from human understanding. Computers struggle with:

- Ambiguity
- Cultural context
- Sarcasm and humor
- Semantic understanding

# Transformers, what can they do?

## Capabilities

Transformer models can solve tasks across multiple modalities, including:

- Natural language processing
- Computer vision
- Audio processing
- Multimodal applications

## Pipeline Function

The `pipeline()` function in the 🤗 Transformers library connects a model with preprocessing and postprocessing steps:

1. Text is preprocessed into model-compatible format
2. Preprocessed inputs are passed to the model
3. Model predictions are post-processed for human interpretation

## Modalities

### Text Pipelines

- **Text generation**: Create content from prompts
- **Text classification**: Categorize text (including zero-shot)
- **Summarization**: Condense text while preserving key information
- **Translation**: Convert between languages
- **Named entity recognition**: Identify persons, locations, organizations
- **Question answering**: Extract answers from context
- **Fill-mask**: Complete sentences with missing words

### Image Pipelines

- **Image-to-text**: Generate descriptions of images
- **Image classification**: Identify objects in images
- **Object detection**: Locate and identify objects

### Audio Pipelines

- **Speech recognition**: Convert speech to text
- **Audio classification**: Categorize audio
- **Text-to-speech**: Convert text to spoken audio

### Multimodal Pipelines

- **Image-text-to-text**: Respond to images based on text prompts

# How do Transformers work?

## History

- **June 2017**: Original Transformer architecture introduced, focused on translation
- **June 2018**: GPT, first pretrained Transformer model
- **October 2018**: BERT, optimized for sentence understanding
- **February 2019**: GPT-2, larger version with better capabilities
- **October 2019**: T5, multi-task sequence-to-sequence model
- **May 2020**: GPT-3, capable of zero-shot learning
- **January 2022**: InstructGPT, trained to follow instructions
- **January 2023**: Llama, multilingual text generation
- **March 2023**: Mistral, efficient 7B model with grouped-query attention
- **May 2024**: Gemma 2, lightweight open models (2B-27B)
- **November 2024**: SmolLM2, compact models (135M-1.7B) for edge devices

## Transformer Architecture

Transformer models generally have two main components:

1. **Encoder**: Processes input to build representation/understanding
2. **Decoder**: Uses encoder's representation to generate output

These components can be used in three primary architectures:

- **Encoder-only models** (like BERT): Good for understanding tasks (classification, NER)
- **Decoder-only models** (like GPT): Good for generative tasks (text generation)
- **Encoder-decoder models** (like T5): Good for tasks requiring both input understanding and output generation (translation, summarization)

## Attention Mechanism

The core innovation of Transformers is the attention mechanism:

- Allows the model to focus on specific words when processing each word
- Captures contextual relationships between words regardless of their distance
- Different types: self-attention, masked attention (can't see future words), cross-attention (between encoder and decoder)

For example, in translation, attention helps the model focus on relevant source words when generating each target word, accounting for grammar differences between languages.

## Transfer Learning

Transformers use a two-stage approach:

1. **Pretraining**: Initial training on massive text datasets (expensive)

   - Models learn language statistics and patterns
   - Requires weeks of training on specialized hardware
   - Significant computational and environmental cost

2. **Fine-tuning**: Adapting pretrained models to specific tasks (efficient)
   - Requires much less data than training from scratch
   - Lower time, financial, and environmental costs
   - Achieves better results than training from scratch

This approach allows knowledge transfer from general language understanding to specific applications, making powerful NLP accessible to more users.
