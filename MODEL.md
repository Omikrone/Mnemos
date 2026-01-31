# Mnemos-0.1.0 Model Configuration and Results

Mnemos-0.1.0 is a mini language model based on the Transformer architecture, designed for educational and experimental purposes. Here are the details about its architecture, components, and performance.

## Architecture & Components

Mnemos is based on the Transformer architecture and consists of several key components:
- **Tokenizer**: A custom BPE ([Byte Pair Encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding)) tokenizer that can handle text encoding and decoding in subword units.
- **Multi-Heads Self-Attention**: A simplified multi-head attention mechanism that allows the model to focus on different parts of the input text (with causal masking).
- **Feed Forward MLP**: A multi-layer perceptron that processes the output from the attention mechanism.
- **Layer Normalization**: A layer normalization component that stabilizes the training process and avoids vanishing or exploding gradients.
- **Transformer Block**: A combination of the attention head, feed forward MLP, and layer normalization, forming the core of the Transformer architecture.
- **Loss Function**: A loss function based on the cross-entropy loss, which is commonly used in language modeling tasks.
- **Gradient Descent**: An optimizer that updates the model parameters based on the computed gradients for the training.
- **Dropout**: A regularization technique to prevent overfitting during training.

## Model Hyperparameter Configuration

The Mnemos-0.1.0 model was trained with the following hyperparameters:
- **Vocabulary Size**: 1024 tokens
- **Maximum Sequence Length**: 64 tokens
- **Embedding Dimension**: 256
- **Hidden Dimension**: 1024
- **Number of Attention Heads**: 4
- **Number of Transformer Blocks**: 4
- **Dropout Rate**: 0.1
- **Initial Learning Rate**: 0.001

With these configurations, I achieved pretty good results on a small dataset.

## Training Dataset

The model was trained on an open-source text dataset available [here](https://huggingface.co/datasets/OpenLLM-France/Claire-Dialogue-French-0.1/blob/main/FR/AssembleeNationale_13/train.txt). The dataset consists of approximately 226 MB (that I truncated to 100 MB) of French text data, and it was used to train the [Claire](https://huggingface.co/OpenLLM-France/Claire-7B-0.1) language model.

## Tests and Results

The first objective of this project is to be a pedagogical tool, so it is not meant to be a production-ready model. With the current configuration and training on the small dataset, the model achieved the following results:
- **Average Training and Test Loss**: 1.6

- **Example of prompt and generated text from current 0.1.0 version**:
```markdown
Prompt: "tout le monde convient "
Generated Text: "la rédotion exculière des consulières mations, souv'hui le sujet de la du garde soixe conter l'une résoluge d'asse. ce de présente sur le prenez ce du proposée par la loi que permettrative en transgité des portères de comptendire des parleurs d'instrieurs mettre qu'il de l'aut"
```

- **Example of prompt and generated text from previous test version**:
```markdown
Prompt: "tout le monde convient "
Generated Text: " l'avient du ss'oscaissant court de l'acfesion donnation suir celaire d'unes deurs deurs mitéréen de la dévenome crecelle sertimenant qu'étrage le sitanduonon le crécele parles en la majours, la comment dom."
```

As you can see, the model is now able to generate text that reproduce the structure of syllables and phrases, but it is not yet coherent or meaningful. This is expected for a small dataset and a simple model architecture. However, the current version shows significant improvement compared to the previous test version.


## Current Limitations

Mnemos is a minimal implementation of a Transformer-based language model, and it is not yet capable of generating coherent or meaningful text. The main limitations are:
- **Small Dataset**: The model was trained on a small dataset, which limits its ability to learn complex patterns and relationships in the data.
- **Simplified Architecture**: The model uses a simplified architecture with only one attention head and a single feed forward MLP, which limits its capacity to learn complex representations.
- **General performance**: The model is not yet optimized for performance, and it may not scale well to larger datasets or more complex tasks.

## Roadmap

The following improvements are planned for the next version:

- **Add more attention heads**: Using multiple attention heads will allow the model to focus on different parts of the input text and learn more complex representations.
- **Increase the model size**: Adding more layers and parameters will increase the model's capacity to learn complex patterns and relationships in the data.
- **Optimize the training process**: Implementing more advanced optimization techniques, such as learning rate scheduling and gradient clipping, will improve the training stability and convergence.
- **Train on better datasets**: Using larger and more diverse datasets will help the model learn a wider range of language patterns and improve its generalization capabilities.
