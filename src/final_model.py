import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

class TransitionPredictor:
    
    def __init__(self, threshold=0.65):
        """
        Initialise le prédicteur.
        
        Args:
            threshold: Seuil de décision pour classification (défaut: 0.65)
        """
        self.model = None
        self.threshold = threshold
        self.feature_names = [
            'Return',
            'Volatility', 
            'Cumulated_Return_5d',
            'RSI14',
            'Volume_ROC',
            'ATR',
            'VIX_spike',
            'Distance_GC',
            'MA_velocity',
            'MA50_slope',
            'Distance_normalized',
            'MA_cross_momentum'
        ]
        self.is_fitted = False
    
    def fit(self, X, y):
        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing Features: {missing_features}")
        
        self.model = RandomForestClassifier(
            n_estimators=100,  
            max_depth=5,
            class_weight='balanced',
            random_state=42,
        )
        
        
        self.model.fit(X, y)  
        self.is_fitted = True
        
        return self
    
    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Call fit first, untrained model")
        
        return self.model.predict_proba(X)[:, 1]  
    
    def predict(self, X):
        """
        Prédit les transitions avec le threshold optimisé.
        
        Args:
            X: DataFrame avec les features
            
        Returns:
            Array de prédictions (0 ou 1)
        """
        proba = self.predict_proba(X)
        return (proba >= self.threshold).astype(int)
    
    def evaluate(self, X, y_true, verbose=True):
        """
        Évalue les performances du modèle.
        
        Args:
            X: DataFrame avec les features
            y_true: Series avec les vraies valeurs
            verbose: Afficher les résultats détaillés
            
        Returns:
            Dict avec les métriques (precision, recall, f1)
        """
        y_pred = self.predict(X)
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'threshold': self.threshold
        }
        
        if verbose:
            print("\n" + "="*50)
            print(f"📊 ÉVALUATION (threshold={self.threshold})")
            print("="*50)
            print(f"Precision: {precision:.3f}")
            print(f"Recall:    {recall:.3f}")
            print(f"F1 Score:  {f1:.3f}")
            print("="*50)
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred, 
                                       target_names=['No Transition', 'Transition']))
        
        return metrics
    
    def get_feature_importance(self, top_n=None):
        
        if not self.is_fitted:
            raise ValueError("Model not yet trained.")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        if top_n:
            importance_df = importance_df.head(top_n)
        
        return importance_df
    
    def tune_threshold(self, X, y_true, thresholds=None):
        
        if thresholds is None:
            thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
        
        y_proba = self.predict_proba(X)
        
        results = []
        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            
            if y_pred.sum() > 0:  # Éviter division par 0
                p = precision_score(y_true, y_pred, zero_division=0)
                r = recall_score(y_true, y_pred, zero_division=0)
                f = f1_score(y_true, y_pred, zero_division=0)
            else:
                p, r, f = 0, 0, 0
            
            results.append({
                'threshold': thresh,
                'precision': p,
                'recall': r,
                'f1': f
            })
        
        results_df = pd.DataFrame(results)
        
        print("\n🎯 Threshold Tuning Results:")
        print(results_df.to_string(index=False))
        
        best_idx = results_df['f1'].idxmax()
        best_thresh = results_df.loc[best_idx, 'threshold']
        print(f"\n✨ Best threshold: {best_thresh} (F1={results_df.loc[best_idx, 'f1']:.3f})")
        
        return results_df
    
    def save(self, filepath):
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"✅ Modèle sauvegardé: {filepath}")
    
    @staticmethod
    def load(filepath):
        
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Modèle chargé: {filepath}")
        return model
    
    def __repr__(self):
        status = "fitted" if self.is_fitted else "not fitted"
        return f"TransitionPredictor(threshold={self.threshold}, status={status})"