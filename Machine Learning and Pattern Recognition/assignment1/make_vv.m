function vv = make_vv(Phi)
    phi1 = ones(size(Phi, 2), 1);
    vv = Phi * inv(Phi.' * Phi).' * phi1;
end