function nada = get_mean_and_std(yy, fig_name)

    % Calcualte standard error.
    stderr = std(yy) / sqrt(size(yy, 1));
    
    % Plot histogram with mean and standard error.
    hist(yy, sqrt(size(yy, 1)));
    title(strcat('\mu = ' + string(mean(yy)) + ' \pm ' + stderr));
    xlabel('CT scan slice location');
    
    saveas(gcf, fig_name);
    
end