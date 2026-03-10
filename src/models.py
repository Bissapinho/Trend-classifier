from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np


class RFCModel:

    def __init__(self, 
            n_estimators=346,
            max_depth=9,
            min_samples_split=17,
            min_samples_leaf=5,
            max_features=0.5463,
        ):
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight='balanced',
            random_state=42
        )
        self.model = RandomForestClassifier(**self.params)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_

class XGBModel:

    def __init__(self, 
            n_estimators=138,
            max_depth=5,
            learning_rate=0.0430,
            subsample=0.7273,
            colsample_bytree=0.6976,
            gamma=0.0536,
            min_child_weight=1,
            eval_metric='logloss',
            verbosity=0
        ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.eval_metric = eval_metric
        self.verbosity = verbosity
        self.model = None

    def fit(self, X_train, y_train):
        spw = (y_train == 0).sum() / (y_train == 1).sum()
        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            scale_pos_weight=spw,
            eval_metric=self.eval_metric,
            random_state=42,
            verbosity=self.verbosity
        )
        self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_