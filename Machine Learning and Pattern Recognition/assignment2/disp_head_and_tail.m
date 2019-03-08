function nada = disp_head_and_tail(console_output, num_lines)
    char_per_line = 48;
    disp('first ' + string(num_lines) + ' lines:');
    console_output(1:(num_lines*char_per_line))
    disp(' ');
    disp('last ' + string(num_lines) + ' lines:');
    console_output(end-(num_lines*char_per_line):end)
end