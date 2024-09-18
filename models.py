import os
import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.metrics import make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from sklearn.metrics import roc_curve

from skopt import BayesSearchCV

from utils import create_bootstrap_samples

from sklearn.model_selection import train_test_split



###########################################################################################################

class BootstrapModel():
    def __init__(self, x_data, y_data, scaler, model, scoring, random_state=None):
        # pandas型をnumpy型に変える（numpy型, pandas型以外はエラーを吐く）
        if isinstance(x_data, pd.DataFrame) or isinstance(x_data, pd.Series):
            x_data = x_data.to_numpy()
        elif not isinstance(x_data, np.ndarray):
            raise TypeError("x_data must be a pandas DataFrame/Series or a numpy ndarray")
        if isinstance(y_data, pd.DataFrame) or isinstance(y_data, pd.Series):
            y_data = y_data.to_numpy()
        elif not isinstance(y_data, np.ndarray):
            raise TypeError("y_data must be a pandas DataFrame/Series or a numpy ndarray")

        # train, valに分割する
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(x_data,  y_data, random_state = random_state)
        print(self.x_train.shape, self.x_val.shape, self.y_train.shape, self.y_val.shape)

        # モデルのインスタンスを作成
        self.pipe = make_pipeline(scaler, model)

        # 評価指標を保持
        self.roc_auc_scorer = make_scorer(scoring)
        self.scoring = scoring

        # レコード用のテキストファイル名
        self.record_file_name = f"{scaler.__class__.__name__}_{model.__class__.__name__}_{scoring.__name__}"


    # ブートストラップサンプルデータを作る
    def samples(self, num_samples, true_sample_size, balance):
        self.num_samples = num_samples
        self.true_sample_size = true_sample_size
        self.balance = balance
        self.bstrap_sample_dict = {}
        create_bootstrap_samples(self.x_train, self.y_train, num_samples, true_sample_size, balance, self.bstrap_sample_dict)


    # 学習
    def train(self, search_spaces, cv, n_iter):
        # 現在の日付を取得
        self.current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 学習モデルとスコアの保管
        self.bstrap_models_dict = {}
        self.record_cv_score = []

        for i in tqdm(range(self.num_samples)):
            X_bstrap = self.bstrap_sample_dict[f'X_bstrap_{i}']
            y_bstrap = self.bstrap_sample_dict[f'y_bstrap_{i}']

            bstrap_model = clone(self.pipe)

            bstrap_grid = BayesSearchCV(bstrap_model, search_spaces=search_spaces, n_jobs=-1, cv=cv, n_iter=n_iter,  scoring=self.roc_auc_scorer)

            bstrap_grid.fit(X_bstrap, y_bstrap)

            #print(f'X_bstrap_{i} SCORE: ', bstrap_grid.best_score_)
            #print(f'X_bstrap_{i} PARAM: ', bstrap_grid.best_params_)

            self.record_cv_score.append(bstrap_grid.best_score_)
            self.bstrap_models_dict[f'bstrap_model_{i}'] = bstrap_grid

        print(f'bstrap_train_score  min：', min(self.record_cv_score))
        print(f'bstrap_train_score  max：', max(self.record_cv_score))
        print(f'bstrap_train_score  ave：', sum(self.record_cv_score) / len(self.record_cv_score))



    # valデータ推論
    def val_pred(self, num):
        # 推論値の保管
        self.bstrap_pred_dict = {}
        self.val_record_score = []

        for i in range(num):
            bstrap_model = self.bstrap_models_dict[f'bstrap_model_{i}']
            bstrap_pred = bstrap_model.predict_proba(self.x_val)[:,1]
            self.bstrap_pred_dict[f'bstrap_pred_{i}'] = bstrap_pred
            #print(f'bstrap_pred_{i} val_Counter：', Counter(bstrap_pred))
            #print(f'bstrap_pred_{i} val_SCORE：', self.scoring(self.y_val, bstrap_pred))
            score = self.scoring(self.y_val, bstrap_pred)
            self.val_record_score.append(score)

        self.ensemble_pred = np.zeros(self.x_val.shape[0])
        for i in range(self.x_val.shape[0]):
            pred_list = []
            for j in range(num):
                pred_list.append(self.bstrap_pred_dict[f'bstrap_pred_{j}'][i])
            # 確率値の平均を計算
            avg_proba = np.mean(pred_list, axis=0)
            self.ensemble_pred[i] = avg_proba

        print(f'bstrap_val_score  min：', min(self.val_record_score))
        print(f'bstrap_val_score  max：', max(self.val_record_score))
        print(f'bstrap_val_score  ave：', sum(self.val_record_score) / len(self.val_record_score))
        print('-------------------------------------------------------------------------------------------------')
        #print(f'ensemble_pred val_Counter：', Counter(self.ensemble_pred))
        print(f'ensemble_pred val_SCORE：', self.scoring(self.y_val, self.ensemble_pred))
        #print('混合行列')
        #print(confusion_matrix(self.y_val, self.ensemble_pred))
        fpr, tpr, thresholds = roc_curve(self.y_val, self.ensemble_pred)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr,tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.show()


    # テストデータ推論
    def test_pred(self, x_test, num, download=False):
        # pandas型をnumpy型に変える（numpy型, pandas型以外はエラーを吐く）
        if isinstance(x_test, pd.DataFrame) or isinstance(x_test, pd.Series):
            x_test = x_test.to_numpy()
        elif not isinstance(x_test, np.ndarray):
            raise TypeError("x_test must be a pandas DataFrame/Series or a numpy ndarray")

        # テスト値の保管
        self.test_pred_dict = {}

        for i in range(num):
            bstrap_model = self.bstrap_models_dict[f'bstrap_model_{i}']
            bstrap_pred = bstrap_model.predict_proba(x_test)[:,1]
            self.test_pred_dict[f'bstrap_pred_{i}'] = bstrap_pred
            #print(f'bstrap_pred_{i} val_Counter：', Counter(bstrap_pred))

        self.test_ensemble_pred = np.zeros(x_test.shape[0])
        for i in range(x_test.shape[0]):
            pred_list = []
            for j in range(num):
                pred_list.append(self.test_pred_dict[f'bstrap_pred_{j}'][i])
            # 確率値の平均を計算
            avg_proba = np.mean(pred_list, axis=0)
            self.test_ensemble_pred[i] = avg_proba

        if download is True:
            # 予測結果をCSVファイルに保存
            current_time = datetime.datetime.now().strftime("%d_%H%M%S")
            file_name = f'../submissions/pred_{current_time}.csv'
            predictions_df = pd.DataFrame(self.test_ensemble_pred, columns=['Prediction'])
            predictions_df.to_csv(file_name, index=False, header=None)

        return self.test_ensemble_pred

    # レコードを保存する関数
    def record(self):
        record_dir = '../record'
        if not os.path.exists(record_dir):
            os.makedirs(record_dir)

        # ファイル名を生成
        file_path = os.path.join(record_dir, f"{self.record_file_name}.txt")

        with open(file_path, 'a') as f:
            f.write("\n===========================================================================================\n")
            f.write(f"\ntrain start: {self.current_time}\n")
            f.write(f"\nsamples(num_samples={self.num_samples}, true_sample_size={self.true_sample_size}, balance={self.balance})\n\n")
            f.write(f"bstrap_train_score  min: {min(self.record_cv_score)}\n")
            f.write(f"bstrap_train_score  max: {max(self.record_cv_score)}\n")
            f.write(f"bstrap_train_score  ave: {sum(self.record_cv_score) / len(self.record_cv_score)}\n")
            f.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
            f.write(f"bstrap_val_score  min: {min(self.val_record_score)}\n")
            f.write(f"bstrap_val_score  max: {max(self.val_record_score)}\n")
            f.write(f"bstrap_val_score  ave: {sum(self.val_record_score) / len(self.val_record_score)}\n")
            f.write(f"ensemble_pred val_SCORE: {self.scoring(self.y_val, self.ensemble_pred)}\n")



