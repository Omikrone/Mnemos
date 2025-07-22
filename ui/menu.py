from generating.inference import Inference
from training.test import Tester
from training.train import Trainer


class UserInterface:
    """ User interface for the Transformer model training and inference. """        


    def run(self):
        """ Run the user interface. """

        print("Welcome! My name is Mnemos, and I am a Transformer model.")
        print("You can train me on text data, test my performance, or generate text based on a prompt.")

        while True:
            print("\nPlease choose an option:")
            print("1. Train me")
            print("2. Test me")
            print("3. Generate text")
            print("4. Exit")

            choice = input("Enter your choice (1-4): ")

            print()
            if choice == "1":
                self.train_model()
            elif choice == "2":
                self.test_model()
            elif choice == "3":
                self.generate_text()
            elif choice == "4":
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please try again.")


    def train_model(self):
        """ Train a new model. """

        self.trainer = Trainer()
        self.trainer.train()


    def test_model(self):
        """ Test the trained model. """

        self.tester = Tester()
        self.tester.test()


    def generate_text(self):
        """ Generate text using the trained model. """

        self.inference = Inference()
        prompt = input("Enter a prompt: ")
        generated_text = self.inference.generate(prompt, max_length=100)
        print(f"Generated text: {generated_text}")