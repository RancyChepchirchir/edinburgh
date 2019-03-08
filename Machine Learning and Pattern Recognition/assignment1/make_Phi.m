function Phi = make_Phi(C, K, tt)
    Phi = zeros(C, K);
    tt_short = tt((20 - C + 1):20);
    for k = 1:K
        Phi(:, k) = tt_short.^(k-1);
    end
end