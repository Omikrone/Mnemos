from mnemos.generating.inference import Inference

class LLM:

    def __init__(self, model: Inference):
        self.model = model

    def generate_text(self, prompt: str) -> str:
        """Generate text based on the given prompt using the model."""
        response = self.model.generate(prompt)
        return response