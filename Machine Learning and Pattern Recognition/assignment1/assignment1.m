
% Load data
load('/afs/inf.ed.ac.uk/group/teaching/mlprdata/audio/amp_data.mat')
% load('amp_data.mat')


% %%%%%
% Question 1
% %%%%%

% Plot amplitude data.
plot(amp_data);
saveas(gcf, 'Q1x_amplitudes.png');
hist(amp_data, 100);
saveas(gcf, 'Q1x_amplitude_hist.png');

% Reshape data to wider form.
col_size = 21;
C = floor(size(amp_data, 1) / col_size);
amp_data = amp_data(1:(C * col_size));  % remove values that would produce an incomplete row
amp_data = reshape(amp_data, col_size, C).';  % reshape is column-wise, so need to switch dims and then transpose

% Get rows for train, val and test sets.
rng(287364823);  % for consistency of results
amp_data = amp_data(randperm(size(amp_data, 1)), :);  % shuffle rows
train_rows = 1:floor(0.7 * size(amp_data, 1));
val_rows = (max(train_rows) + 1):floor(0.85 * size(amp_data, 1));
test_rows = (max(val_rows) + 1):size(amp_data, 1);

% Get columns for X and y.
x_ids = 1:(col_size - 1);
y_ids = col_size;

% Separate into train, val and test sets.
X_shuf_train = amp_data(train_rows, x_ids);
y_shuf_train = amp_data(train_rows, y_ids);
X_shuf_val = amp_data(val_rows, x_ids);
y_shuf_val = amp_data(val_rows, y_ids);
X_shuf_test = amp_data(test_rows, x_ids);
y_shuf_test = amp_data(test_rows, y_ids);


% %%%%%
% Question 2
% %%%%%

% Part a
% %%%%%

% Plot one row of x and y values in training set.
tt = (0:(1/20):(19/20)).';
row_to_plot = 2;  % just chose this because it looked nice; others looked nice too
hold off;  % just in case I'm running code non-linearly...
plot(tt, X_shuf_train(row_to_plot, :));
hold on;
plot(1, y_shuf_train(row_to_plot), '*');  % plot t = 1 value as asterisk
saveas(gcf, 'Q2a_one_row.png');

% Part b
% %%%%%

% Fit straight line to the 20 points.
Phi_1b = [ones(20, 1), tt];
w_fit = Phi_1b \ X_shuf_train(row_to_plot, :).';
ff = Phi_1b * w_fit;
plot(tt, ff);
saveas(gcf, 'Q2b_line_through_points.png');

% Part c
% %%%%%

% Fit quadratic polynomial.
Phi_1c = [ones(20, 1), tt, tt.^2, tt.^3, tt.^4];
w_fit = Phi_1c \ X_shuf_train(row_to_plot, :).';
ff = Phi_1c * w_fit;
plot(tt, ff);
saveas(gcf, 'Q2c_polynomial_through_points.png');


% %%%%%
% Question 3
% %%%%%

% Part b
% %%%%%

% Construct C x K design matrix Phi.
Phi = make_Phi(5, 3, tt);  % first argument is C, second is K
vv = make_vv(Phi);

% Compare predictions to poly fit from before.
for X_train_row = 1:4
    
    w_fit = Phi_1c \ X_shuf_train(X_train_row, :).';
    phi1 = ones(size(Phi_1c, 2), 1);
    prediction1 = w_fit.' * phi1;
    
    vv = make_vv(make_Phi(20, 5, tt));  % C = 20, K = 5
    prediction2 = X_shuf_train(X_train_row, :) * vv;
    
    disp(strcat('row: ' + string(X_train_row)));
    disp(strcat('w^T * phi(t=1): ' + string(prediction1)));
    disp(strcat('v^T * x: ' + string(prediction2)));
    disp(' ')
end

% Part c
% %%%%%

% Evaluate various C and K. Not very efficient to calculate all test results, but I find
% it cleaner.
C_max = 20;  % will test C from 1 to C_max
K_max = 5;  % will test K from K to min(C, K_max) (since I want K <= C)
E_train = Inf * ones(C_max, K_max);  % initialize with Inf b/c will take min
E_val = Inf * ones(C_max, K_max);
E_test = Inf * ones(C_max, K_max);
for C = 1:C_max
    
    % Create data snippets of length C.
    C_idx = (20 - C + 1):20;
    X_shuf_train_C = X_shuf_train(:, C_idx);
    X_shuf_val_C = X_shuf_val(:, C_idx);
    X_shuf_test_C = X_shuf_test(:, C_idx);
    
    % For each K, create v and record error for training and validation.
    for K = 1:min(C, K_max)
        vv = make_vv(make_Phi(C, K, tt));
        E_train(C, K) = mean((X_shuf_train_C * vv - y_shuf_train).^2);
        E_val(C, K) = mean((X_shuf_val_C * vv - y_shuf_val).^2);
        E_test(C, K) = mean((X_shuf_test_C * vv - y_shuf_test).^2);
    end
end

print_best_CK_and_error(E_train, 'train');
print_best_CK_and_error(E_val, 'val');

% Get test error value for best C, K.
test_error_3c = E_test(2, 2);
disp(strcat('Test error for optimal C, K: ' + string(test_error_3c)));


% %%%%%
% Question 4
% %%%%%

% Part a
% %%%%%

% Evaluate various C.
C_max = 20;
E_train = Inf * ones(C_max, 1);  % initialize error array
E_val = Inf * ones(C_max, 1);
for C = 1:C_max
    
    C_idx = (20 - C + 1):20;
    
    % Fit weights using training data.
    w_fit = X_shuf_train(:, C_idx) \ y_shuf_train;
    
    % Apply weights to both training and validation data.
    ff_train = X_shuf_train(:, C_idx) * w_fit;
    ff_val = X_shuf_val(:, C_idx) * w_fit;
    
    % Calculate error for training and validation error.
    E_train(C) = mean((ff_train - y_shuf_train).^2);
    E_val(C) = mean((ff_val - y_shuf_val).^2);
    
end

% get test error and best C in training set
[M, C] = min(E_train);

% Get test error for best C in validation set.
[M, C] = min(E_val);  % get best C
test_error_4b = E_test(C);


% Part b
% %%%%%

% Sloppy code to perform sanity check: look at some plots and predictions.
num_plots = 9;
start_row = 250;
plot_num = 1;
grid_width = floor(sqrt(num_plots));

hold off;
for row_to_plot = start_row:(start_row + num_plots - 1)
    
    subplot(grid_width, grid_width, plot_num);

    % Pick row to plot and C
    C = 18;
    C_idx = (20 - C + 1):20;
    N_train = size(X_shuf_train, 1);
    % N_val = size(X_shuf_val, 1);

    % Find fit on training and get prediction for t = 1.
    w_fit = [ones(N_train, 1), X_shuf_train(:, C_idx)] \ y_shuf_train;
    ff_val = [1, X_shuf_val(row_to_plot, C_idx)] * w_fit;

    % Plot whole curve (including t = 1).
    hold off;
    plot([tt(C_idx); 1], [X_shuf_val(row_to_plot, C_idx), y_shuf_val(row_to_plot)]);

    % Plot the prediction where we fitted v.
    hold on;
    plot(1, ff_val, '*');

    % Plot polynomial prediction.
    C = 2;
    K = 2;
    C_idx = (20 - C + 1):20;
    vv = make_vv(make_Phi(C, K, tt));
    ff_val2 = X_shuf_val(row_to_plot, C_idx) * vv;
    plot(1, ff_val2, '+');
    legend('data', 'fitted v', 'poly');
    title(strcat('validation row: ' + string(row_to_plot)))

    plot_num = plot_num + 1;
end
saveas(gcf, 'Q4b_grid_plot_sanity_check.png');

% Plot training error and validation error against C.
hold off;
plot(1:size(E_train, 1), E_train);
hold on;
plot(1:size(E_train, 1), E_val);
title('Training & validation error vs. C');
xlabel('C');
ylabel('error');
legend('E_{train}','E_{val}');
saveas(gcf, 'Q4b_training_vs_error.png');

% Compare to best C to best polynomial from question 3c.
test_error_4b / test_error_3c


% Part d
% %%%%%

% Plot hist of residuals on validation data 

% Calculate residuals.
C = 2;
K = 2;
C_idx = (20 - C + 1):20;
X_shuf_val_C = X_shuf_val(:, C_idx);
vv = make_vv(make_Phi(C, K, tt));
residuals = X_shuf_val_C * vv - y_shuf_val;

% Plot residuals.
hist(residuals, 100);
saveas(gcf, 'Q4c_residuals_hist.png');
2 * std(residuals);
