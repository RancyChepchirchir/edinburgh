function nada = print_best_C_and_error(E, set_type)

    [M, I] = min(E(:));
    C = ind2sub(size(E), I);
    disp(strcat('Best C in ' + string(set_type) + ' set:'));
    disp(strcat('C = ' + string(C) + ' (error = ' + string(M) + ')'));
    disp(' ');
end