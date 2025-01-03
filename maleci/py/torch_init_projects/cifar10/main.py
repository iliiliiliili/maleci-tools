CODE = '''import torch
from fire import Fire
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from models.cnn import MnistCnnClassifier
from models.mlp import MnistMlpClassifier


def create_dataloaders():
    train_data = datasets.MNIST(
        root="data",
        train=True,
        transform=ToTensor(),
        download=True,
    )
    test_data = datasets.MNIST(root="data", train=False, transform=ToTensor())

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=100, shuffle=True, num_workers=1
    )

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=100, shuffle=True, num_workers=1
    )

    return train_loader, test_loader


def train(num_epochs, model, train_loader, loss_func, optimizer):

    model.train()

    # Train the model
    total_step = len(train_loader)

    for epoch in range(num_epochs):
        # For each batch in the training data
        for i, (images, labels) in enumerate(train_loader):

            # Compute output and loss
            output = model(images)
            loss = loss_func(output, labels)

            # Clear gradients for this training step
            optimizer.zero_grad()

            # Compute gradients
            loss.backward()
            # Apply gradients
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, num_epochs, i + 1, total_step, loss.item()
                    )
                )

    print("Done.")


def test(model, test_loader):
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            test_output = model(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            correct += (pred_y == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / float(total)
        print("Test Accuracy: %.2f%%" % (accuracy * 100))


def main(model_name="cnn", learning_rate=0.01, epochs=10, **kwargs):
    train_loader, test_loader = create_dataloaders()

    model = {
        "cnn": MnistCnnClassifier,
        "mlp": MnistMlpClassifier,
    }[
        model_name
    ](**kwargs)

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train(epochs, model, train_loader, loss_func, optimizer)
    test(model, test_loader)


if __name__ == "__main__":
    Fire(main)
'''