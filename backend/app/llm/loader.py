from app.llm.model import LLM

from mnemos.generating.inference import Inference

_llm_instance = None

def load_llm() -> LLM:
    global _llm_instance

    if _llm_instance is None:

        model = Inference()
        _llm_instance = LLM(model=model)

    return _llm_instance
