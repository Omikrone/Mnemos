from generating.inference import Inference
from training.test import Tester
from training.train import Trainer



#trainer = Trainer()
#trainer.train()
#tester = Tester()
#tester.test()
#exit(0)

if __name__ == "__main__":
    inference = Inference()
    prompt = "Comment ça va ? "
    generated_text = inference.generate(prompt, max_length=50)
    print(f"Prompt: {prompt}")
    print(f"Generated text: {generated_text}")