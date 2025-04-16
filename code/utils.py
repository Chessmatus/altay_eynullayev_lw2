import numpy as np
from tSSA import t_SSA
from sklearn.model_selection import train_test_split

def optimal_cpd_rank(train_data, val_data, w_size, random_state=42):
    cpd_ranks = np.arange(5, 30 + 1, 5)
    forecast_results = {key: {"mse": None, "mape": None} for key in cpd_ranks}

    # parameter for detereministic behaviour of tSSA
    random_state = 42
    cpd_errors = []
    mape_forecast = []

    for cpd_rank in cpd_ranks:
        print(f"CP rank = {cpd_rank}")
        t_ssa_obj = t_SSA(w_size, train_data.T, cpd_rank)

        # make svd for common matrix, extract factors and singular values
        t_ssa_obj.decompose_tt(random_state=random_state)
        cpd_errors.append(t_ssa_obj.cpd_err_rel)
        print(f"CPD-error = {t_ssa_obj.cpd_err_rel}")

        t_ssa_obj.remove_last_predictions()

        # get prediction for cuurent number of factors left
        forecast_tssa = np.empty(val_data.shape)

        for i in range(val_data.shape[0]):
            forecast_tssa[i] = np.array(t_ssa_obj.predict_next())

        # get MSE for every signal
        signals_mse_tssa = np.mean((forecast_tssa - val_data) ** 2, axis=0)
        # get MAPE for every signal
        signals_mape_tssa = np.mean(np.abs((forecast_tssa - val_data) / val_data), axis=0)

        forecast_results[cpd_rank]["mse"] = signals_mse_tssa
        forecast_results[cpd_rank]["mape"] = signals_mape_tssa
        mape_forecast.append(np.mean(signals_mape_tssa))

        print(f'MSE: {signals_mse_tssa}; Mean by signals = {np.mean(signals_mse_tssa):e}')
        print(f'MAPE: {signals_mape_tssa}; Mean by signals = {np.mean(signals_mape_tssa):e}')

    return forecast_results, cpd_errors, mape_forecast