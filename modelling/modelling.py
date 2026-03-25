from model.randomforest import RandomForest


def model_predict(data, df, name):
    print(f"\n{'='*60}")
    print(f"  Model: RandomForest  |  Run: '{name}'")
    print(f"{'='*60}")
    model = RandomForest(
        model_name=name,
        embeddings=data.get_embeddings(),
        y=data.get_type()
    )
    model.train(data)
    model.predict(data.get_X_test())
    model_evaluate(model, data)
    return model


def model_evaluate(model, data):
    model.print_results(data)
