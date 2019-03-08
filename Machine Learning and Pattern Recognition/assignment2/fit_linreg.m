function [ww, bb] = fit_linreg(X, yy, alpha)
    
    % Add row of ones to data.
    N = size(X, 1);
    Phi = [ones(N, 1), X];

    % Create regularized data.
    K = size(Phi, 2);
    Phi_reg_addon = sqrt(alpha) * eye(K);
    Phi_reg_addon(1, 1) = 0;  % easy way to make sure we don't regularize the bias term
    Phi_reg = [Phi; Phi_reg_addon];
    y_reg = [yy; zeros(K, 1)];
    
    % Fit weights and get fitted y values. Predict using ff = Phi * w_fit.
    w_fit = Phi_reg \ y_reg;
    bb = w_fit(1);
    ww = w_fit(2:end);

end

