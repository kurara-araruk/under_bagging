import numpy as np

# ブートストラップサンプリング関数
def create_bootstrap_samples(X_train, y_train, num_samples=10, sample_size=1000, balance=1, dict_name=None):
    if dict_name is None:
        dict_name = {}
    bstrap_dict = dict_name

    where_is_0 = np.where(y_train == 0)[0].tolist()
    where_is_1 = np.where(y_train == 1)[0].tolist()

    for i in range(num_samples):
        kimini_kimeta_0 = np.random.choice(where_is_0, size=int(balance*sample_size), replace=True)
        kimini_kimeta_1 = np.random.choice(where_is_1, size=sample_size, replace=True)

        X_0_sample = X_train[kimini_kimeta_0]
        y_0_sample = y_train[kimini_kimeta_0]

        X_1_sample = X_train[kimini_kimeta_1]
        y_1_sample = y_train[kimini_kimeta_1]

        X_bstrap = np.concatenate([X_0_sample, X_1_sample])
        y_bstrap = np.concatenate([y_0_sample, y_1_sample])
        bstrap_indices = np.arange(X_bstrap.shape[0])
        np.random.shuffle(bstrap_indices)
        X_bstrap = X_bstrap[bstrap_indices]
        y_bstrap = y_bstrap[bstrap_indices]

        bstrap_dict[f'X_bstrap_{i}'] = X_bstrap
        bstrap_dict[f'y_bstrap_{i}'] = y_bstrap

    return bstrap_dict
