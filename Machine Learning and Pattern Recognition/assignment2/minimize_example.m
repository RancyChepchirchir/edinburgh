% Load X_test, X_train, X_val, y_test, y_train, y_val.
load('/afs/inf.ed.ac.uk/group/teaching/mlprdata/ctslice/ct_data.mat')

% Remove bad columns.
cols_to_drop_idx = max(X_train) - min(X_train) == 0;
X_train = X_train(:, ~cols_to_drop_idx);
X_val = X_val(:, ~cols_to_drop_idx);
X_test = X_test(:, ~cols_to_drop_idx);

% Dimension reminders:
% ww Kx1 hidden-output weights
% bb 1x1 output bias
% V KxD hidden-input weights
% bk Kx1 hidden biases

% Initialize weights.
K = 10;
D = size(X_train, 2);

ww_init = 0.1 * randn(K, 1) / sqrt(K);
bb_init = randn(1) / sqrt(K);
V_init = 0.1 * randn(K, D) / sqrt(D);
bk_init = randn(K, 1) / sqrt(D);

init = {ww_init, bb_init, V_init, bk_init};

% Run minimization.
alpha = 10;
num_line_searches = 500;

param = minimize(init, @nn_cost, -num_line_searches, X_train, y_train, alpha);  % only get one function evaluation.

ww = param{1};
bb = param{2};
V = param{3};
bk = param{4};

