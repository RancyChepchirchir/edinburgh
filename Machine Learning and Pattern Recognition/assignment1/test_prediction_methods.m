function is_equal = test_prediction_methods_(tt, Phi_1c, X_shuf_train, shuf_row)
    
    w_fit = Phi_1c\X_shuf_train(shuf_row, :).';
    phi1 = ones(size(Phi_1c, 2), 1);
    prediction1 = w_fit.' * phi1;

    vv = make_vv(make_Phi(20, 5, tt));  % C = 20, K = 5
    prediction2 = vv.' * X_shuf_train(shuf_row, :).';

    disp(strcat('row: ' + string(shuf_row)));
    disp(strcat('w^T * phi(t=1)): ' + string(prediction1)));
    disp(strcat('v^T * x): ' + string(prediction2)));
    disp('')
end