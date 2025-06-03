import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def display(data, y):
    """
    Display images with their predicted classes.
    :param data: Image data (tensor or numpy array)
    :param y: Predicted classes
    """
    img_num = min(len(data), 32)
    c, h, w = data[0].shape
    plt.figure(figsize=(img_num / 4, img_num / 2))
    for i in range(img_num):
        plt.subplot(8, 4, i + 1)
        # Display the image and the predicted class as the title
        plt.imshow(data[i].reshape(h, w, c), cmap='gray')
        plt.title(f'predicted class: {y[i].item()}')
        plt.axis('off')

    # Save the figure before displaying
    plt.savefig('../results.png', bbox_inches='tight')
    plt.show()

def display_stats(training_stats):
    """
    Display training and validation statistics (accuracy and loss) over epochs.
    :param training_stats: Dictionary containing the training statistics
    """
    epochs = np.arange(1, len(training_stats["train_acc"]) + 1)  # Create an array for epochs

    # Convert stats to a pandas DataFrame for easier plotting with seaborn
    import pandas as pd
    stats_df = pd.DataFrame({
        'Epoch': epochs,
        'Train Accuracy': training_stats['train_acc'],
        'Validation Accuracy': training_stats['valid_acc'],
        'Train Loss': training_stats['train_losses'],
        'Validation Loss': training_stats['valid_losses']
    })

    # Create the accuracy plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Epoch', y='Train Accuracy', data=stats_df, label='Train Accuracy', color='blue')
    sns.lineplot(x='Epoch', y='Validation Accuracy', data=stats_df, label='Validation Accuracy', color='orange')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    plt.show()

    # Create the loss plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Epoch', y='Train Loss', data=stats_df, label='Train Loss', color='blue')
    sns.lineplot(x='Epoch', y='Validation Loss', data=stats_df, label='Validation Loss', color='orange')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper left')
    plt.show()
