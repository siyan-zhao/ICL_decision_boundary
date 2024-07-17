from sklearn.datasets import make_circles, make_moons
import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


def generate_x_y(
    num_samples,
    num_dimensions,
    seed,
    data_type="linear",
    factor=0.5,
    class_sep=1,
    noise_moon=0.05,
    num_classes=2,
):
    """Generate X and y data based on the specified data type."""
    if data_type == "linear":
        X, y = make_classification(
            n_samples=num_samples,
            n_features=num_dimensions,
            n_informative=num_dimensions,
            n_redundant=0,  # no redundant features
            n_clusters_per_class=1,  # each class is a single cluster
            flip_y=0,  # no noise
            shuffle=True,
            random_state=seed,
            n_classes=num_classes,
            class_sep=class_sep,  # make classes clearly separable
        )
    elif data_type == "circle":
        X, y = make_circles(n_samples=num_samples, shuffle=True, noise=0.05, random_state=seed, factor=factor)
    elif data_type == "moon":
        X, y = make_moons(n_samples=num_samples, shuffle=True, noise=noise_moon, random_state=seed)

    # Normalize X to [0, 1] and then scale to [0, 100]
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X = 100 * (X - X_min) / (X_max - X_min)

    return X, y


def generate_tasks(
    num_tasks, num_samples_per_task, num_dimensions, seed, data_type="linear", factor=0.5, class_sep=2
):
    """Generate multiple tasks, each with its own dataset."""
    # Create empty arrays to store X and y data
    X_data = np.zeros((num_tasks, num_samples_per_task, num_dimensions))
    Y_data = np.zeros((num_tasks, num_samples_per_task))

    for i in range(num_tasks):
        X, y = generate_x_y(
            num_samples=num_samples_per_task,
            num_dimensions=num_dimensions,
            seed=seed + i,
            data_type=data_type,
            factor=factor,
            class_sep=class_sep,
        )
        X_data[i] = X
        Y_data[i] = y

    print(f"Generated {num_tasks} tasks with {num_samples_per_task} samples each.")
    return X_data, Y_data


def generate_context_prompt(X, y, class_names):
    y_named = [class_names[int(label)] for label in y]

    prompt = ""
    for features, label in zip(X, y_named):
        features_str = " ".join(f"{int(num)}" for num in np.round(features))
        prompt += f"Input: {features_str}\nLabel: {label}\n"
    return prompt


def generate_dataset(args, meta_train_X, meta_train_y):
    """Generate context and query datasets for training and testing."""
    context_x = []
    context_y = []
    query_x = []
    query_y = []

    for task_idx, (task_x, task_y) in enumerate(zip(meta_train_X, meta_train_y)):
        num_per_class = args.num_in_context // 2 + args.num_test_samples // 2
        class_0_indices = np.where(task_y == 0)[0][:num_per_class]
        class_1_indices = np.where(task_y == 1)[0][:num_per_class]
        context_0_indices = class_0_indices[: args.num_in_context // 2]
        context_1_indices = class_1_indices[: args.num_in_context // 2]
        test_0_indices = class_0_indices[args.num_in_context // 2 :]
        test_1_indices = class_1_indices[args.num_in_context // 2 :]
        context_indices = np.concatenate([context_0_indices, context_1_indices])
        test_indices = np.concatenate([test_0_indices, test_1_indices])
        np.random.shuffle(context_indices)

        context_x.append(task_x[context_indices])
        context_y.append(task_y[context_indices])
        query_x.append(task_x[test_indices])
        query_y.append(task_y[test_indices])

        # Ensure no overlap between context and query sets
        assert len(set(context_indices) & set(test_indices)) == 0

    print("Generated context and query datasets.")
    return np.array(context_x), np.array(context_y), np.array(query_x), np.array(query_y)
