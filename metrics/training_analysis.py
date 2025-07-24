from matplotlib import pyplot as plt
import csv


def plot_training_loss(log_file):
    """ Plot the training loss from the log file. """
    
    epochs = []
    batches = []
    losses = []

    with open(log_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            epoch, batch, loss = map(float, row)
            epochs.append(epoch)
            batches.append(batch)
            losses.append(loss)

    plt.figure(figsize=(10, 5))
    plt.plot(batches, losses, label='Training Loss', color='blue')
    plt.xlabel('Batch Number')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Batches')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    log_file_path = "training_log.csv"
    plot_training_loss(log_file_path)