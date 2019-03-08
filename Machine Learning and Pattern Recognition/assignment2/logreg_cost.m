function [E, params_bar] = logreg_cost(params, X, yy, alpha)
%LOGREG_COST cost function and gradients for regularized logistic regression
%
%     [E, params_bar] = logreg_cost(params, X, yy, alpha)
%
% Can be minimized with minimize.m -- see fit_linreg_gradopt.m for a
% demonstration of how to use this optimizer on a cost function like this one.
%
% Inputs:
%          params cel weights and bias: {ww, bb}, ww Dx1, bb 1x1
%               X NxD design matrix
%              yy Nx1 binary targets in {0,1} or {-1,+1}
%          alpha 1x1 regularization constant
%
% Outputs:
%               E 1x1 regularized cost function value
%      params_bar cel gradients wrt params

% Iain Murray, October 2016

% Unpack parameters from array
ww = params{1};
bb = params{2};

% Force targets to be +/- 1
yy = 2*(yy==1) - 1;

% forward computation of error
aa = yy.*(X*ww + bb);
sigma = 1./(1 + exp(-aa));
E = -sum(log(sigma)) + alpha*(ww'*ww);

% reverse computation of gradients
aa_bar = sigma - 1;
bb_bar = aa_bar'*yy;
ww_bar = X'*(yy.*aa_bar) + 2*alpha*ww;

% put derivatives into one array for minimize.m
params_bar = {ww_bar, bb_bar};

