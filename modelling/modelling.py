# modelling.py
# =============================================================================
# Orchestrates model training, prediction and evaluation.
# Called by main.py -> perform_modelling()
# =============================================================================

from model.randomforest import RandomForest


def model_predict(data, df, name: str):
    """
    Instantiate a RandomForest model, train it, run predictions,
    and print evaluation results.

    Parameters
    ----------
    data : Data
        DataObject containing X_train, y_train, X_test, y_test splits.
    df   : pd.DataFrame
        Full cleaned DataFrame (passed to RandomForest constructor for
        access to embeddings and labels if needed).
    name : str
        A descriptive name / label for this model run (used in output headers).
    """
    print(f"\n{'='*60}")
    print(f"  Model: RandomForest  |  Run: '{name}'")
    print(f"{'='*60}")

    # Instantiate the RandomForest model
    # embeddings = full feature matrix; y = full label series
    model = RandomForest(
        model_name=name,
        embeddings=data.get_embeddings(),
        y=data.get_type()
    )

    # Train on the training split
    model.train(data)

    # Predict on the test split
    model.predict(data.get_X_test())

    # Print accuracy + classification report
    model_evaluate(model, data)

    return model


def model_evaluate(model, data):
    """
    Print evaluation metrics for a trained model.

    Parameters
    ----------
    model : BaseModel subclass
        A trained model with a print_results() method.
    data  : Data
        DataObject whose y_test is used for evaluation.
    """
    model.print_results(data)
