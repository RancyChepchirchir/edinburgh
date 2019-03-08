
%% Question 1

% %%%%%
% Part a
% %%%%%

% Load X_test, X_train, X_val, y_test, y_train, y_val.
load('/afs/inf.ed.ac.uk/group/teaching/mlprdata/ctslice/ct_data.mat')

% Print histograms with means and standard errors.
get_mean_and_std(y_train, 'Q1a_y_train_histogram.png');
get_mean_and_std(y_val, 'Q1a_y_val_histogram.png');
get_mean_and_std(y_train(1:5785), 'Q1a_y_train_short_histogram.png');

% Plot time series of y_train.
plot(y_train);
xlabel('Observation number');
ylabel('CT scan slice location');
saveas(gcf, 'Q1a_y_train_time_series.png');


% %%%%%
% Part b
% %%%%%

cols_to_drop_idx = max(X_train) - min(X_train) == 0;
cols_to_drop = find(cols_to_drop_idx);
disp('Columns to drop:')
for col_to_drop = cols_to_drop
    disp(string(col_to_drop));
end

X_train = X_train(:, ~cols_to_drop_idx);
X_val = X_val(:, ~cols_to_drop_idx);
X_test = X_test(:, ~cols_to_drop_idx);


%% Question 2

alpha = 10;

% Fit weights using least squares operator.
[ww1, bb1] = fit_linreg(Phi, y_train, alpha);
ff1 = Phi * [bb1; ww1];

% Create histogram.
hist(ff1 - y_train, sqrt(size(y_train, 1)))
xlabel('Prediction minus observed');
saveas(gcf, 'Q2x_regularized_error_hist.png')

% Show scatter plot of prediction vs. observed value.
scatter(y_train, ff1)
xlabel('Observed');
ylabel('Predicted');
saveas(gcf, 'Q2x_regularized_error_scatter.png');

[ww2, bb2] = fit_linreg_gradopt(X_train, y_train, alpha);
ff2 = Phi * [bb2; ww2];

% Compare weights.
scatter(ww1, ww2);
xlabel('Fitted weights using least squares operator');
ylabel('Fitted weights using gradient opt');
saveas(gcf, 'Q2x_weight_scatter.png');

% Calculate errors for both sets of weights.
calc_error = @(ff, yy) sum((ff - yy).^2);

err_train1 = calc_error(ff1, y_train);
err_train2 = calc_error(ff2, y_train);

err_val1 = calc_error(ff1, y_train);
err_val2 = calc_error(ff2, y_train);

% X_tmp = [10 11 12 13 14 14 14]';
% y_tmp = [1 2 3 4 5 4 4]';
% Phi = make_Phi(X_tmp);
% % [ww, bb] = fit_linreg_gradopt(X_tmp, y_tmp, 10);
% [ww, bb] = fit_linreg(Phi, y_tmp, 10);
% ff = Phi * [bb; ww];
% hold off;
% scatter(X_tmp, y_tmp);
% hold on;
% plot(Phi(:, 2), ff);

X_mu = mean(X_train, 1); % Matlab/Octave