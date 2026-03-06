import numpy as np
import shap
import tensorflow as tf
from typing import List, Dict

# Disable eager execution for SHAP if using older TF versions, 
# but for TF 2.x DeepExplainer works best with a background dataset.

class AirQualityExplainer:
    def __init__(self, model_path: str, background_data: np.ndarray, feature_names: List[str]):
        """
        model_path: Path to the saved GRU model (.h5 or .keras)
        background_data: A representative sample of the training data (scaled) 
                         used to initialize the SHAP explainer.
        feature_names: List of feature names strings.
        """
        self.model = tf.keras.models.load_model(model_path)
        self.feature_names = feature_names
        # Using GradientExplainer or DeepExplainer for Keras models
        self.explainer = shap.GradientExplainer(self.model, background_data)

    def explain_prediction(self, input_sequence: np.ndarray) -> List[Dict[str, float]]:
        """
        input_sequence: (1, window_size, num_features) scaled input
        returns: List of feature contributions for the prediction
        """
        # shap_values is a list for multi-output models (PM2.5, PM10, O3, SO2)
        shap_values = self.explainer.shap_values(input_sequence)
        
        explanations = []
        target_names = ['PM2.5', 'PM10', 'O3', 'SO2']
        
        for i, target in enumerate(target_names):
            # Summing SHAP values over the time window to get global feature importance for this specific prediction
            # shap_values[i] shape is (1, window_size, features)
            importance = np.sum(shap_values[i][0], axis=0) 
            
            # Format into a dictionary of feature -> weight
            feat_imp = {self.feature_names[j]: float(importance[j]) for j in range(len(self.feature_names))}
            
            # Sort by absolute impact
            sorted_imp = dict(sorted(feat_imp.items(), key=lambda item: abs(item[1]), reverse=True))
            
            explanations.append({
                "pollutant": target,
                "impacts": sorted_imp
            })
            
        return explanations

    def get_top_reasons(self, input_sequence: np.ndarray, top_n: int = 3) -> Dict[str, List[str]]:
        """
        Returns human-readable reasons for each pollutant's prediction.
        """
        raw_exps = self.explain_prediction(input_sequence)
        reasons = {}
        
        for exp in raw_exps:
            pollutant = exp['pollutant']
            impacts = exp['impacts']
            
            pollutant_reasons = []
            for feat, val in list(impacts.items())[:top_n]:
                direction = "high" if val > 0 else "low"
                pollutant_reasons.append(f"{feat} ({direction} impact)")
            
            reasons[pollutant] = pollutant_reasons
            
        return reasons
