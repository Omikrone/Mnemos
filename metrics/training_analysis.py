from matplotlib import pyplot as plt
import csv



def plot_training_loss(training_log_file, validation_log_file):
    """ Plot the training and validation loss from the log files. """

    # --- Lecture du log d'entraînement ---
    train_batches = []
    train_losses = []
    with open(training_log_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            _, batch, loss, _ = map(float, row)
            train_batches.append(batch)
            train_losses.append(loss)

    # --- Lecture du log de validation ---
    val_batches = []
    val_losses = []
    with open(validation_log_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            _, batch, loss, _ = map(float, row)
            val_batches.append(batch)
            val_losses.append(loss)

    # --- Affichage ---
    plt.figure(figsize=(10, 5))
    plt.plot(train_batches, train_losses, label='Training Loss', color='blue')
    plt.plot(val_batches, val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Batch Number')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    training_log_file_path = "logs/training_log.csv"
    validation_log_file_path = "logs/validation_log.csv"
    plot_training_loss(training_log_file_path, validation_log_file_path)