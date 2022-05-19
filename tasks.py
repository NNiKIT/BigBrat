import warnings

from invoke import task

warnings.filterwarnings("ignore")


@task
def train(ctx):
    import torch

    from classification_model.initialize_model import initialize_model
    from classification_model.trainer import Trainer

    config = ctx.classification_model
    model, input_size = initialize_model(
        config.model_name, config.n_classes, config.feature_extract, config.use_pretrained
    )
    if config.model_name == "inception":
        model.aux_logits = False
    config["IMG_SIZE"] = [input_size, input_size]
    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    trainer = Trainer(config, model, opt, criterion)

    history = trainer.train()


@task
def inference(ctx, image_path):
    # from efficientnet_pytorch import EfficientNet
    from classification_model.initialize_model import initialize_model
    from classification_model.trainer import Trainer

    config = ctx.classification_model

    opt = None
    criterion = None
    model = initialize_model_model(
        config.n_classes, config.feature_extract, config.use_pretrained
    )  # .to(DEVICE)

    trainer = Trainer(config, model, opt, criterion)

    image_class = trainer.inference(image_path)
    print('Image "{}" has a class "{}"'.format(image_path.split("/")[-1], image_class))


@task
def get_confusion_matrix(ctx):
    from classification_model.plots import plot_confusion_matrix
    from classification_model.trainer import Trainer, initialize_model_model

    config = ctx.classification_model

    opt = None
    criterion = None
    model = initialize_model_model(
        config.n_classes, config.feature_extract, config.use_pretrained
    )  # .to(DEVICE)

    trainer = Trainer(config, model, opt, criterion)

    print("Get matrix")
    mode = "office"
    confusion_mtx = trainer.get_confusion_matrix(mode)
    path_to_save = "./pics/confusion_matrix_{}.png".format(mode)
    plot_confusion_matrix(
        confusion_mtx,
        classes=[0, 1],
        title="Confusion matrix:" + mode,
        normalize=False,
        save_path=path_to_save,
    )
    print('Confusion matrix is saved to "{}".'.format(path_to_save))


@task
def get_scores(ctx):
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, recall_score

    from classification_model.trainer import Trainer, initialize_model_model

    config = ctx.classification_model

    opt = None
    criterion = None
    model = initialize_model_model(
        config.model_name, config.n_classes, config.feature_extract, config.use_pretrained
    )  # .to(DEVICE)

    mode = "edinburgh"
    trainer = Trainer(config, model, opt, criterion)
    y_true, probs = trainer.predict(mode)

    y_pred = np.argmax(probs, axis=1)
    print("Accuracy: ", accuracy_score(y_true, y_pred))
    print("F1-score: ", f1_score(y_true, y_pred))
    print("Recall-score: ", recall_score(y_true, y_pred))
