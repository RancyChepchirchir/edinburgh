function E_rmse = rmse(E_vector)
    E_rmse = sqrt(mean((E_vector).^2));
end