function [ww, bb] = fit_logreg_gradopt(X, yy, alpha)
    D = size(X, 2);
    num_line_searches = 500;
    init = {zeros(D,1), 0};
    param = minimize(init, @logreg_cost, -num_line_searches, X, yy, alpha);
    ww = param{1};
    bb = param{2};
end


