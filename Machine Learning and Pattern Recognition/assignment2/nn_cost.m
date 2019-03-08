function [E, params_bar] = nn_cost(params, X, yy, alpha)
%NN_COST simple neural network cost function and gradients, or predictions
%
%     [E, params_bar] = nn_cost({ww, bb, V, bk}, X, yy)
%                pred = nn_cost({ww, bb, V, bk}, X)
%
% Cost function E can be minimized with minimize.m
%
% Inputs:
%         params cel {ww, bb, V, bk}, where:
%                --------------------------------
%                    ww Kx1 hidden-output weights
%                    bb 1x1 output bias
%                     V KxD hidden-input weights
%                    bk Kx1 hidden biases
%                --------------------------------
%              X NxD input design matrix
%             yy Nx1 regression targets
%         alpha 1x1 regularization for weights
%
% Outputs:
%              E 1x1 sum of squares error
%     params_bar cel gradients wrt params
% OR
%           pred Nx1 predictions if only params and X are given as inputs

% Iain Murray, October 2016

ww = params{1};
bb = params{2};
V = params{3};
bk = params{4};

% Forwards computation of cost
A = bsxfun(@plus, X*V', bk(:)'); % NxK
P = 1 ./ (1 + exp(-A)); % NxK
F = P*ww + bb; % Nx1
if nargin == 2
    % if user omits yy, assume they want predictions rather than training signal:
    E = F;
    return
end
res = F - yy; % Nx1
E = (res'*res) + alpha*(V(:)'*V(:) + ww'*ww); % 1x1

% Reverse computation of gradients
if nargout > 1
    F_bar = 2*res; % Nx1
    ww_bar = P'*F_bar + 2*alpha*ww; % Kx1
    bb_bar = sum(F_bar); % 1x1
    P_bar = F_bar * ww'; % NxK
    A_bar = P_bar .* P .* (1 - P); % NxK
    V_bar = A_bar' * X + 2*alpha*V; % KxD
    bk_bar = sum(A_bar, 1)';
    
    params_bar = {ww_bar, bb_bar, V_bar, bk_bar};
end

