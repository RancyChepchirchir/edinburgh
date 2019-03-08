function [ff, ww, bb, V, bk] = pred_nn(X_train, y_train, X_val, alpha, init)

    [ww, bb, V, bk] = fit_nn_gradopt(X_train, y_train, alpha, init);
    ff = nn_cost({ww, bb, V, bk}, X_val);

end