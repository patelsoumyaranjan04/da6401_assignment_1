
import argparse
import numpy as np
import json
import os
import wandb
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset


def parse_arguments():

    parser = argparse.ArgumentParser(description="Train MLP")

    parser.add_argument("-d", "--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion_mnist"])

    parser.add_argument("-e", "--epochs", type=int, default=30)

    parser.add_argument("-b", "--batch_size", type=int, default=32)

    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)

    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0001)

    parser.add_argument("-o", "--optimizer", type=str, default="rmsprop",
                        choices=["sgd", "momentum", "nag", "rmsprop"])

    parser.add_argument("-nhl", "--num_layers", type=int, default=3)

    parser.add_argument("-sz", "--hidden_size", nargs="+",type=int,
                    default=[128,128,64])
    parser.add_argument("-a", "--activation", type=str, default="relu",
                        choices=["sigmoid", "tanh", "relu"])

    parser.add_argument("-l", "--loss", type=str, default="cross_entropy",
                        choices=["mse", "mean_squared_error", "cross_entropy"])

    parser.add_argument("-w_i", "--weight_init", type=str, default="xavier",
                        choices=["random", "xavier"])

    parser.add_argument("-w_p","--wandb_project",default="da6401_assignment_1_my-src")

    parser.add_argument("--model_save_path", type=str,
                        default="best_model.npy")

    parser.add_argument("--config_save_path", type=str,
                        default="best_config.json")
    parser.add_argument("--no_wandb",action="store_true")

    parser.add_argument("--experiment", type=str,required=False,
                    help="Name of experiment for W&B logging")
    return parser.parse_args()


def main():

    args = parse_arguments()
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))

   
    args.model_save_path = os.path.join(SRC_DIR, args.model_save_path)
    args.config_save_path = os.path.join(SRC_DIR, args.config_save_path)
    # parsed_hidden = []

    # for h in args.hidden_size:
    #     if isinstance(h, str) and " " in h:
    #         parsed_hidden.extend([int(x) for x in h.split()])
    #     else:
    #         parsed_hidden.append(int(h))

    # args.hidden_size = parsed_hidden
    # args.num_layers = len(args.hidden_size)
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=args.experiment,
        tags=[args.dataset] if args.experiment is None else [args.experiment, args.dataset]
        )

        config = wandb.config

        args.dataset = config.dataset
        args.epochs = config.epochs
        args.batch_size = config.batch_size
        args.learning_rate = config.learning_rate
        args.optimizer = config.optimizer
        args.activation = config.activation
        args.hidden_size = config.hidden_size
        args.num_layers = config.num_layers
        args.weight_decay = config.weight_decay
        args.weight_init = config.weight_init



    # Load dataset
    X_train, y_train, X_test, y_test = load_dataset(args.dataset)
    if not args.no_wandb and args.experiment == "data_exploration":

        table = wandb.Table(columns=["image", "label"])

        raw_images = X_train.reshape(-1, 28, 28)
        raw_labels = np.argmax(y_train, axis=1)

        for label in range(10):
            indices = np.where(raw_labels == label)[0][:5]

            for idx in indices:
                table.add_data(wandb.Image(raw_images[idx]), label)

        wandb.log({"data_exploration_samples": table})

    X=X_train
    y=y_train
    val_ratio=0.1
    # Train/validation split
    num_samples = X.shape[0]
    num_val = int(num_samples * val_ratio)
    
    # Random permutation
    indices = np.random.permutation(num_samples)
    val_indices = indices[:num_val]
    train_indices = indices[num_val:]
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_val = X[val_indices]
    y_val = y[val_indices]


    print("Training samples:", X_train.shape[0])
    print("Validation samples:", X_val.shape[0])
    print("Test samples:", X_test.shape[0])

    # Initialize model
    model = NeuralNetwork(args)

    # print("\nModel Architecture:")
    # for i, layer in enumerate(model.layers):
    #     w_shape = layer.W.shape if hasattr(layer, "W") else None
    #     b_shape = layer.b.shape if hasattr(layer, "b") else None

    #     print(
    #         f"Layer {i}: "
    #         f"Weights {w_shape}, "
    #         f"Bias {b_shape}, "
    #         f"Activation {getattr(layer, 'activation', None)}"
    #     )
    # print()


    from sklearn.metrics import f1_score

    print("\ntraining...\n")

    for epoch in range(args.epochs):

        train_loss, train_acc = model.train_epoch(
            X_train, y_train, args.batch_size
        )

 
        val_loss, val_acc, val_preds = model.evaluate(X_val, y_val)

        val_targets = np.argmax(y_val, axis=1)

        val_f1 = f1_score(val_targets, val_preds, average="weighted")

        print(
            f"Epoch {epoch+1}/{args.epochs} "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_acc:.4f} "
            f"Val F1: {val_f1:.4f}"
        )

        if not args.no_wandb:
            wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_f1": val_f1
    })


    final_weights = model.get_weights()
    model.set_weights(final_weights)

    test_loss, test_acc, test_preds = model.evaluate(X_test, y_test)

    test_targets = np.argmax(y_test, axis=1)

    from sklearn.metrics import precision_score, recall_score

    test_f1 = f1_score(test_targets, test_preds, average="weighted")
    test_precision = precision_score(test_targets, test_preds,
                                     average="weighted",
                                     zero_division=0)
    test_recall = recall_score(test_targets, test_preds,
                               average="weighted",
                               zero_division=0)

    print("Test Accuracy:", test_acc)
    print("Test F1:", test_f1)
    print("Test Precision:", test_precision)
    print("Test Recall:", test_recall)

    if not args.no_wandb:
        wandb.log({
            "test_accuracy": test_acc,
            "test_f1": test_f1,
            "test_precision": test_precision,
            "test_recall":test_recall
        })

    # Save model
    np.save(args.model_save_path, final_weights)
    print(f"Model saved to {args.model_save_path}")

    # Save config
    config = vars(args)
    config['test_f1'] = float(test_f1)
    config['test_accuracy'] = float(test_acc)

    with open(args.config_save_path, "w") as f:
        json.dump(config, f, indent=2)

    print("Training complete!")

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()