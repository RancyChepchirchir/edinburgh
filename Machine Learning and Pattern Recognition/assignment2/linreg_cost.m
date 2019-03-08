function [E, param_bar] = linreg_cost(param, X, yy, alpha)
%LINREG_COST cost function and gradients for linear regression
%
%     [E, param_bar] = linreg_cost(param, X, yy, alpha)
%
% E = sum(((X*ww + bb) - yy).^2) + alpha*ww'*ww;
% param_bar contains the gradients wrt {ww,bb}, and can be used with the
% optimizer minimize.m -- see fit_linreg_gradopt.m for a demonstration.
%
% Inputs:
%         param cel {ww, bb}: Dx1 weights ww, 1x1 bias bb
%             X NxD design matrix of input features
%            yy Nx1 real-valued targets 
%         alpha 1x1 regularization constant
%
% Outputs:
%             E 1x1 scalar regularized cost
%     param_bar cel {ww_bar, bb_bar}

% Iain Murray, October 2016

% Unpack parameters from array
ww = param{1};
bb = param{2};

% forward computation of error
ff = X*ww + bb;
res = (ff - yy);
E = res'*res + alpha*(ww'*ww);

% reverse computation of gradients
ff_bar = 2*res;
bb_bar = sum(ff_bar);
ww_bar = X'*ff_bar + 2*alpha*ww;

% put derivatives into one array for minimize.m
param_bar = {ww_bar, bb_bar};
