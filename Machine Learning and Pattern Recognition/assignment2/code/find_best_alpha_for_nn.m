function [alpha_idx, E_train, E_val] = find_best_alpha_for_nn(init, data, alphas, make_plot)

    % Unpack data.
    X_train = data{1};
    y_train = data{2};
    X_val = data{3};
    y_val = data{4};
    
    % Initialize error vectors.
    E_val = zeros(size(alphas));
    E_train = zeros(size(alphas));
    alpha_idx = 1;
    
    for alpha = alphas
        
        disp('testing alpha ' + string(alpha) + '...');
        
        [console_output, ww, bb, V, bk] = evalc('fit_nn_gradopt(X_train, y_train, alpha, init);');

        % Calculate RMSE for training.
        ff = nn_cost({ww, bb, V, bk}, X_train);
        E_train(alpha_idx) = rmse(ff - y_train);

        % Calculate RMSE for validation.
        ff = nn_cost({ww, bb, V, bk}, X_val);
        E_val(alpha_idx) = rmse(ff - y_val);

        alpha_idx = alpha_idx + 1;

    end

    % Get alpha value with lowest validation error and report errors for
    % that alpha.
    [M, alpha_idx] = min(E_val);
    alpha_best = alphas(alpha_idx);
    disp(' ');
    disp('best alpha:                      ' + string(alpha_best));
    disp('training error for best alpha:   ' + string(E_train(alpha_idx)));
    disp('validation error for best alpha: ' + string(E_val(alpha_idx)));
    
    if make_plot
    
        % Create plot for training and validation errors by alpha.
        clf;
        semilogx(alphas, E_train, 'b');
        hold on;
        semilogx(alphas, E_val, 'r');
        title('RMSE by regularization constant \alpha');
        legend('Training', 'Validation');
        xlabel('\alpha');
        ylabel('RMSE');
        hold off;
    
    end

end