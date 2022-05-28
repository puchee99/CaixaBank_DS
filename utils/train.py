from sklearn.preprocessing import StandardScaler
import pandas as pd 
import numpy as np
import scipy 
import seaborn as sns
import matplotlib.pyplot as plt
class Utils(object):

    def func_overX(self, X):
        Y = []
        for element in X:
            Y.append(sum(element.flatten()) > element.flatten().shape[0] // 2)
        return np.asarray(Y)

    def _ensure_dimensionalit(self, arr):
        return arr if len(arr[0].shape) == 1 else [x.flatten() for x in arr]

    def _acc(self, y_pred, y_target):

        if type(y_pred) == np.array and type(y_target) == np.array:
            assert(y_pred.shape == y_target.shape)
            mask = y_pred == y_target

        else:
            assert(len(y_pred) == len(y_target))
            mask = [x == y for x, y in zip(y_pred, y_target)]
        return sum(mask)/len(mask)

    def do_scaling(self, X):
        Scaler = StandardScaler()
        return Scaler.fit_transform(X)

    def _gen_gridSearch(self, model, hyperparams, n_splits=5):
        cv = StratifiedShuffleSplit(
            n_splits=n_splits, test_size=0.2, random_state=42)
        grid = GridSearchCV(model, param_grid=hyperparams,
                            cv=cv, n_jobs=6, verbose=0)
        return grid

    def df_Grid(self):
        if self.grid_flag:
            c = self.grid.__dict__['cv_results_']['params']
            a = ['params'] + \
                [f'split{n}_test_score' for n in range(self.n_splits)]
            data = pd.DataFrame({h: i for h, i in zip(
                a, (c, *[self.grid.__dict__['cv_results_'][f'split{n}_test_score'] for n in range(self.n_splits)]))})
            
            data[list(data['params'][0].keys())] = pd.DataFrame(data['params'].tolist())
            
            return data
        else:
            print('Grid has not been calculated')
                       
    def ci(self, alpha):
        def f(x):
            return scipy.stats.t.interval(alpha = alpha, df = self.n_splits - 1, loc = x['mean'], scale = x['sem'])
        return f

    def top_params(self, alpha = 0.95, n = None): #retorna els parametres amb millor ci acc
        df = self.df_Grid()
        df['mean'] = df.filter(regex='test').mean(axis = 1) #agafem columnes nombrades 'split*' calculem mitja
        df['sem'] = df.filter(regex='test').apply(scipy.stats.sem, axis = 1) + 1e-8 #standard error of mean
        df['ci'] = df.apply(self.ci(alpha), axis = 1)
        df['sort'] = [0.5 * x[1] - abs(x[0] - x[1]) * 0.5 for x in df['ci']] 
        df = df.sort_values('sort', ascending=False)
        
        if n:
            return df[:n]
        return df[:]

    def boxplots(self, n_params=10, duplicates=False):  # for top_params_df
        df = self.top_params(0.95, n_params)
        
        df['name'] = df[list(pd.DataFrame(df['params'].tolist()))].astype(str).agg('-'.join, axis=1)
        df = df.loc[:, df.columns.str.contains('score|name')].set_index('name')
        if not duplicates:
            df.drop_duplicates(inplace = True)
        df = pd.DataFrame(pd.DataFrame(df.unstack('name'), columns=[
                        'value']).droplevel(0)).reset_index(level=0)

        sns.set(font='Gill Sans', font_scale=1.2,
                palette='pastel', style="whitegrid")

        fig = plt.figure(figsize=(13, 6))
        ax = fig.add_subplot(111)

        # Plot with horizontal boxes
        sns.boxplot(x='value', y='name', data=df, width=0.6)

        # Tweak the visual presentation
        ax.xaxis.grid(True)
        ax.set(xlabel="Accuracy", ylabel="")
        sns.despine(trim=True, left=True, bottom=True)
        plt.title(f"{n_params} Splits Boxplots {str(self.model.__class__).split('.')[-1][:-2]}")
        plt.show()


from sklearn import svm
from sklearn import neighbors
from sklearn import tree
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
import pickle

class ModelHandler(Utils):

    models = {'SVM': svm.SVC, 'KNN': neighbors.KNeighborsClassifier,
              'DT': tree.DecisionTreeClassifier, 'XGB': XGBClassifier}
    hyperparams = {'SVM': {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': np.logspace(-2, 10, 3),
    }, 'KNN': {
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'n_neighbors': np.arange(3, 10, 2),
        'p': np.arange(1, 3),
    }, 'DT': {
        'criterion': ['gini', 'entropy'],
        'max_features': ['auto', 'sqrt', 'log2'],
        'splitter': ['best', 'random'],
    }, 'XGB': {
    	'booster': ['gbtree', 'gblinear'],
        'min_child_weight': [0.5, 1, 3, 5, 10],
        'gamma': [0.5, 1, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.5, 1.0, 2.0],
        'max_depth': [4, 26, 32, 64 ]
    }}

    def __init__(self, X, Y, model: str, **kwargs):

        super().__init__()

        self.n_splits = kwargs['n_splits'] if 'n_splits' in kwargs else 5
        if 'n_splits' in kwargs:
            del kwargs['n_splits']
        self.hyperparam = self.hyperparams[model]
        self.hyperparam.update(kwargs)
        self.model = self.models[model](
            **{x: kwargs[x][0] if type(kwargs[x]) == list else kwargs[x] for x in kwargs})
        self.X = self.do_scaling(self._ensure_dimensionalit(X))
        self.Y = Y
        assert(len(Y.shape) == 1)
        self.grid = self._gen_gridSearch(
            self.model, self.hyperparam, self.n_splits)
        self.grid_flag = False

    def fit(self, with_score=True, with_grid=True):
        if with_grid:
            self.grid_flag = True
            self.grid.fit(self.X, self.Y)
            print(f"[INFO] The best parameters are {self.grid.best_params_}")
            print(f"[INFO] The best score is {self.grid.best_score_:.4f}")
            top_vals = self.top_params(0.95, 1).params.values[0]
            print(f"[INFO] The best parameters according to ci are {top_vals}")
            self.model = self.model.__class__(**top_vals)
            self.model.fit(self.X, self.Y)
        else:
            self.model = self.model.fit(self.X, self.Y)
        if with_score:
            pred = self.predict(self.X)
            print(f"[INFO] Train acc  is : {self._acc(pred, self.Y):.4f}")

    def predict(self, X):
        X = self._ensure_dimensionalit(X)
        return self.model.predict(self.do_scaling(X))

    def available_models(self):
        return self.models.keys()
    
    def save(self, name = False):
        if not name:
            name = str(self.model.__class__).split('.')[-1][:-2]
        pickle.dump(self.model, open(name + '.pickle','wb'))
    
    def load(self,path):
        self.model = pickle.load(open(path,'rb'))