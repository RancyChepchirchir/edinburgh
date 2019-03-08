function nada = print_best_CK_and_error(E, set_type)
    [M, I] = min(E(:));
    [C, K] = ind2sub(size(E), I);
    disp(strcat('Best C, K in ' + string(set_type) + ' set:'));
    disp(strcat('C = ' + string(C) + ', K = ' + string(K) + ' (error = ' + string(M) + ')'));
    disp(' ');
end