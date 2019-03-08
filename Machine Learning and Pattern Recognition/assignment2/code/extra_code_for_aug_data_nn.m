% Create design matrix including binary values.
aug_fn = @(X) [X X > mean(X)];

X_train_bin = aug_fn(X_train);

D = size(X_train_bin, 2);

V_init_bin = 0.1 * randn(K, D) / sqrt(D);
bk_init_bin = randn(K, 1) / sqrt(D);

init_bin = {ww_init, bb_init, V_init_bin, bk_init_bin};

alpha = 1;
[ww3, bb3, V3, bk3] = fit_nn_gradopt(X_train_bin, y_train, alpha, init_bin);


% Calculate RMSE for training.
ff3 = nn_cost({ww3, bb3, V3, bk3}, aug_fn(X_train));
rmse(ff3 - y_train)

% Calculate RMSE for validation.
ff3 = nn_cost({ww3, bb3, V3, bk3}, aug_fn(X_val));
rmse(ff3 - y_val)