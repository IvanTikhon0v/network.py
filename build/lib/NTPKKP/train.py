import numpy as np


def train(network, x_train, y_train, loss_func, epochs, batch_size=None, optimizer=None):
    n_samples = x_train.shape[0]
    for epoch in range(epochs):
        print(f"\nЭпоха {epoch + 1}/{epochs}:")
        if batch_size:
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            x_train, y_train = x_train[indices], y_train[indices]

            for start_idx in range(0, n_samples, batch_size):
                end_idx = start_idx + batch_size
                x_batch = x_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]

                # Backpropagation and weight update for batch
                network.backward(x_batch, y_batch, loss_func)
                for layer in network.layers:
                    if hasattr(layer, 'update_weights'):
                        layer.update_weights(network.learning_rate, optimizer=optimizer)

                # Compute loss for the current batch
                batch_loss = (loss_func(y_batch, network.forward(x_batch)))/y_batch.size

                # Вывод ошибки каждые 100 батчей
                batch_num = start_idx // batch_size + 1
                if batch_num % 100 == 0:
                    print(f"  Батч {batch_num}, Потеря: {batch_loss:.4f}")

        else:  # Training without batching
            network.backward(x_train, y_train, loss_func)
            for layer in network.layers:
                if hasattr(layer, 'update_weights'):
                    layer.update_weights(network.learning_rate, optimizer=optimizer)

        # Compute loss for the entire epoch
        epoch_loss = (loss_func(y_train, network.forward(x_train)))/y_train.size
        print(f"Итоговая потеря за эпоху: {epoch_loss:.4f}")
