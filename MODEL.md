

## Architecture & Components

Mnemos is based on the Transformer architecture and consists of several key components:
- **Tokenizer**: A custom BPE ([Byte Pair Encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding)) tokenizer that can handle text encoding and decoding in subword units.
- **2-Heads Self-Attention**: A simplified multi-head attention mechanism that allows the model to focus on different parts of the input text (with causal masking).
- **Feed Forward MLP**: A multi-layer perceptron that processes the output from the attention mechanism.
- **Layer Normalization**: A layer normalization component that stabilizes the training process and avoids vanishing or exploding gradients.
- **Transformer Block**: A combination of the attention head, feed forward MLP, and layer normalization, forming the core of the Transformer architecture.
- **Loss Function**: A loss function based on the cross-entropy loss, which is commonly used in language modeling tasks.
- **Gradient Descent**: An optimizer that updates the model parameters based on the computed gradients for the training.

## Tests and Results

The first objective of this project is to be a pedagogical tool, so it is not meant to be a production-ready model. I tested Mnemos on a small dataset of 20 MB, and achieved following results:
- **Average Training and Test Loss**: 1.85

- **Example of prompt and generated text from previous version**:
```markdown
Prompt : "tout le monde convient "
Generated text : " avioncont de qulem'onast eronsore le a de ra"
```

- **Example of prompt and generated text from current version**:
```markdown
Prompt: "tout le monde convient "
Generated Text: " l'avient du ss'oscaissant court de l'acfesion donnation suir celaire d'unes deurs deurs mitéréen de la dévenome crecelle sertimenant qu'étrage le sitanduonon le crécele parles en la majours, la comment dom."
```

As you can see, the model is able to generate text that reproduce the structure of syllables and phrases, but it is not yet coherent or meaningful. This is expected for a small dataset and a simple model architecture.

This shows that the model is able to generate text that resembles the structure of the input text, but it is still not coherent or meaningful. The model is still in its early stages of development and requires further training and improvements to achieve better results.


## Limitations and Future Work

Mnemos is a minimal implementation of a Transformer-based language model, and it is not yet capable of generating coherent or meaningful text. The main limitations are:
- **Small Dataset**: The model was trained on a small dataset, which limits its ability to learn complex patterns and relationships in the data.
- **Simplified Architecture**: The model uses a simplified architecture with only one attention head and a single feed forward MLP, which limits its capacity to learn complex representations.
- **General performance**: The model is not yet optimized for performance, and it may not scale well to larger datasets or more complex tasks.

The following improvements are planned for the next version:

- [X] **Add more attention heads**: Using multiple attention heads will allow the model to focus on different parts of the input text and learn more complex representations.
- [X] **Increase the model size**: Adding more layers and parameters will increase the model's capacity to learn complex patterns and relationships in the data.
- [ ] **Optimize the training process**: Implementing more advanced optimization techniques, such as learning rate scheduling and gradient clipping, will improve the training stability and convergence.
- [ ] **Improve the user interface**: Enhancing the command-line interface to provide more options and better feedback to the user.
