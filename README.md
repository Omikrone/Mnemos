# Mnemos - Mini-LLM based on Transformers

## Introduction

**Mnemos** (comes from the [Greek deity of memory](https://en.wikipedia.org/wiki/Mnemosyne)) is a mini-LLM based on Transformers, designed for training and testing purposes. It is built to be lightweight and efficient, making it suitable for educational and experimental use.
This pedagogical project is built from scratch, so all the different components are available in this repository, and you don't need to install much additional dependencies.

- **Current Version**: 0.2.0 (POC)

## Features

Mnemos includes the following user features:
- **Training on custom datasets**: You can load your own datasets and process them for training. Mnemos will automatically handle the tokenization and batching of the data, and adjust the model parameters accordingly.
- **Testing with custom datasets**: You can test the model with your own datasets, allowing you to evaluate its performance on a separate validation set.
- **Generation of text**: Once trained, Mnemos can generate text based on the learned patterns from the training data. You can make your own prompts and see how the model responds.


## Installation

To install Mnemos, clone the repository and install the required dependencies:

```bash
git clone https://github.com/Omikrone/Mnemos.git mnemos
cd mnemos
pip install -r requirements.txt
```

Then, you can run the main file to choose between training, testing, or generating text:

```bash
python main.py
```


## Usage

Mnemos is designed to be user-friendly and straightforward. It provides a simple command-line interface to interact with the model. You can choose to train, test, or generate text by running the main script.

### Training Instructions

To train the model, follow these steps:
1. Prepare your dataset in a text file format and rename it to `train.txt`. The size of the training dataset should be between 1 MB and 25 MB for effective training.
2. Place the `train.txt` file in the `data/training/` directory at the root of the project (create this directory if it doesn't exist).
3. Choose the training option in the main menu.

### Testing Instructions

If you want to test the model, follow these steps:
1. Prepare your dataset in a text file format and rename it to `test.txt`.
2. Place the `test.txt` file in the `data/testing/` directory at the root of the project (create this directory if it doesn't exist).
3. Choose the testing option in the main menu.

### Inference Instructions

Once the model is trained, you can use it for inference (text generation) by choosing the inference option in the main menu and providing a prompt.


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
- **Average Training and Test Loss**: 2.5

- **Example of prompt and generated text from previous version**:
```markdown
Prompt: "comment ça va ?"
Generated Text: "nionercensgiscoion nerin lesensseerger"
```

As you can see, the model is able to generate text that reproduce the structure of syllables and phrases, but it is not yet coherent or meaningful. This is expected for a small dataset and a simple model architecture.

- **Example of prompt and generated text from current version**:
```markdown
Prompt : "tout le monde convient "
Generated text : " avioncont de qulem'onast eronsore le a de ra"
```

This shows that the model is able to generate text that resembles the structure of the input text, but it is still not coherent or meaningful. The model is still in its early stages of development and requires further training and improvements to achieve better results.


## Limitations and Future Work

Mnemos is a minimal implementation of a Transformer-based language model, and it is not yet capable of generating coherent or meaningful text. The main limitations are:
- **Small Dataset**: The model was trained on a small dataset, which limits its ability to learn complex patterns and relationships in the data.
- **Simplified Architecture**: The model uses a simplified architecture with only one attention head and a single feed forward MLP, which limits its capacity to learn complex representations.
- **General performance**: The model is not yet optimized for performance, and it may not scale well to larger datasets or more complex tasks.

The following improvements are planned for the next version:

- [X] **Add more attention heads**: Using multiple attention heads will allow the model to focus on different parts of the input text and learn more complex representations.
- [ ] **Increase the model size**: Adding more layers and parameters will increase the model's capacity to learn complex patterns and relationships in the data.
- [ ] **Optimize the training process**: Implementing more advanced optimization techniques, such as learning rate scheduling and gradient clipping, will improve the training stability and convergence.
- [ ] **Improve the user interface**: Enhancing the command-line interface to provide more options and better feedback to the user.
