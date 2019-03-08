function nada = explore_dist(yy, title_str)

    % Calculate standard error.
    stderr = std(yy) / sqrt(size(yy, 1));
    disp(strcat('mean = ' + string(mean(yy)) + ' +/- ' + stderr))
    
    % Plot histogram with mean and standard error.
    hist(yy, sqrt(size(yy, 1)));
    title(title_str);
    xlabel('CT scan slice location');
    
end