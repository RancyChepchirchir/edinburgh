function [ww, bb, V, bk] = fit_nn_gradopt(X, yy, alpha, init)
    num_line_searches = 500;
    param = minimize(init, @nn_cost, -num_line_searches, X, yy, alpha);
    ww = param{1};
    bb = param{2};
    V = param{3};
    bk = param{4};
end