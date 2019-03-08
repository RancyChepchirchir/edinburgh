function [ww, bb] = fit_linreg_gradopt(X, yy, alpha)
%FIT_LINREG_GRADOPT fit a regularized linear regression model with gradient opt
%
%     [ww, bb] = fit_linreg_gradopt(X, yy, alpha)
%
% Find weights and bias by using a gradient-based optimizer (minimize.m) to
% improve the regularized least squares cost:
%   sum(((X*ww + bb) - yy).^2) + alpha*ww'*ww
%
% Inputs:
%         X NxD design matrix of input features
%        yy Nx1 real-valued targets
%     alpha 1x1 regularization constant
%
% Outputs:
%        ww Dx1 fitted weights
%        bb 1x1 fitted bias

% Iain Murray, October 2016

D = size(X, 2);
num_line_searches = 500;
init = {zeros(D,1), 0};
param = minimize(init, @linreg_cost, -num_line_searches, X, yy, alpha);
ww = param{1};
bb = param{2};

