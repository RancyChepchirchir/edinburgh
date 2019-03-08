function V = pca_zm_proj(X, K)
%PCA_ZM_PROJ return PCA projection matrix for zero mean data
%
%     V = pca_zm_proj(X, K)
%
% Inputs:
%     X NxD design matrix of input features -- must be zero mean
%     K 1x1 how many columns to return in projection matrix
%
% Outputs:
%     V DxK matrix to apply to X or other matrices shifted in same way.

% Iain Murray, October 2016

[N, D] = size(X);
if nargin < 2
    K = D;
end

if max(abs(mean(X,1))) > 1e-9
    error('Data is not zero mean.');
end

[V, E] = eig(X'*X);
[E,id] = sort(diag(E),1,'descend');
V = V(:, id(1:K)); % DxK
 
