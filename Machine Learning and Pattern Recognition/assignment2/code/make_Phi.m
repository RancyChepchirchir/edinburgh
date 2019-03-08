function Phi = make_Phi(X)
    N = size(X, 1);
    Phi = [ones(N, 1), X];
end