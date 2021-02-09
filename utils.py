"""
    Receives X_train, Y_train, and X_train as Pandas Dataframe.
    Returns as Dict of Numpy Arrays.
"""
def scikit_learn_glue(x_train, y_train, x_test, y_test, y_train_vol, y_test_vol, features):
    x_train = x_train[features].to_numpy()
    y_train = y_train.to_numpy()

    # Need a way of mapping a y_test example to a prediction date
    permno_dates = {
        'permno': x_test.permno.to_numpy(),
        'date': x_test.date.to_numpy(),
        'prediction_date': x_test.prediction_date.to_numpy()
    }

    x_test = x_test[features].to_numpy()
    y_test = y_test.to_numpy()

    if y_train_vol is not None:
        y_train_vol = y_train_vol.to_numpy()

    if y_test_vol is not None:
        y_test_vol = y_test_vol.to_numpy()

    return (x_train, y_train, x_test, y_test, permno_dates, y_train_vol, y_test_vol)

def lstm_glue(x_train, y_train, x_test, y_test, y_train_vol, y_test_vol, features):
    x_train = x_train[features].to_numpy()
    x_train = x_train.reshape(-1, 1, x_train.shape[1])

    y_train = y_train.to_numpy().reshape(-1, 1)

    # Need a way of mapping a y_test example to a prediction date
    permno_dates = {
        'permno': x_test.permno.to_numpy(),
        'date': x_test.date.to_numpy(),
        'prediction_date': x_test.prediction_date.to_numpy()
    }

    x_test = x_test[features].to_numpy()
    x_test = x_test.reshape(-1, 1, x_test.shape[1])

    y_test = y_test.to_numpy().reshape(-1, 1)

    if y_train_vol is not None:
        y_train_vol = y_train_vol.to_numpy().reshape(-1, 1)

    if y_test_vol is not None:
        y_test_vol = y_test_vol.to_numpy().reshape(-1, 1)

    return (x_train, y_train, x_test, y_test, permno_dates, y_train_vol, y_test_vol)

STANDARD_WEIGHT_CONSTRAINTS =  [
    [(0, 1), (0,1)],
    [(0, 1), (0,0)],
    # For example this constrains stocks to 10% of the portfolio and leaves bonds unconstrained.
    [(0, 0.1), (0, 1)],
    [(0, 0.1), (0, 0.1)],
    [(0, 0.1), (0, 0)],
    [(0, 0.05), (0, 1)],
    [(0, 0.05), (0, 0.05)],
    [(0, 0.05), (0, 0)]
]
