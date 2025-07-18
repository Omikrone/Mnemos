from generating.inference import Inference
from training.test import Tester
from training.train import Trainer


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
    #tester = Tester()
    #tester.test()
    #exit(0)


    inference = Inference()
    prompt = "vous parlez d'une dette de 2, 3 milliards "
    generated_text = inference.generate(prompt, max_length=100)
    print(f"Prompt: {prompt}")
    print(f"Generated text: {generated_text}")