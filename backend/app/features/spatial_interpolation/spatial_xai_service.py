"""
Explainable AI (XAI) service for spatial interpolation.
Generates human-readable explanations for why predictions are made.
"""

import math
from typing import Dict, List, Tuple
import numpy as np


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two points using Haversine formula.
    Returns distance in kilometers.
    """
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
         math.sin(dlon/2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return 6371 * c  # km


def calculate_idw_weights(distances: List[float], power: float = 2.0) -> List[float]:
    """
    Calculate Inverse Distance Weighting (IDW) weights.
    power=2.0 means inverse distance squared.
    Returns normalized weights as percentages (0-100).
    """
    eps = 1e-12
    weights = 1.0 / (np.array(distances) ** power + eps)
    weights_normalized = (weights / weights.sum()) * 100
    return weights_normalized.tolist()


def categorize_scenario(batt_percent: float, kandy_percent: float) -> str:
    """
    Categorize the scenario based on weight distribution.
    
    Returns: "near_station", "leaning", or "middle"
    """
    batt_dominant = batt_percent > kandy_percent
    dominant_percent = max(batt_percent, kandy_percent)
    
    if dominant_percent > 80:
        return "near_station"
    elif dominant_percent > 65:
        return "leaning"
    else:
        return "middle"


def generate_explanation(
    lat: float,
    lon: float,
    target: str,
    batt_lat: float,
    batt_lon: float,
    batt_pred: float,
    kandy_lat: float,
    kandy_lon: float,
    kandy_pred: float,
    final_pred: float
) -> Dict:
    """
    Generate XAI explanation for a prediction point.
    
    Returns dict with:
    - explanation: Primary explanation text
    - extra_line: Additional context (if middle scenario)
    - scenario: Type of scenario (near_station, leaning, middle)
    - distances: Dict of distances to each station
    - weights: Dict of IDW weights (percentages)
    - station_values: Dict of predicted values at each station
    - dominant_station: Which station dominates (if applicable)
    """
    
    # Calculate distances
    distance_batt = haversine_distance(lat, lon, batt_lat, batt_lon)
    distance_kandy = haversine_distance(lat, lon, kandy_lat, kandy_lon)
    distances = [distance_batt, distance_kandy]
    
    # Calculate IDW weights
    weights = calculate_idw_weights(distances, power=2.0)
    batt_percent = weights[0]
    kandy_percent = weights[1]
    
    # Determine scenario
    scenario = categorize_scenario(batt_percent, kandy_percent)
    
    # Determine which station is dominant
    if batt_percent > kandy_percent:
        dominant_station = "Battaramulla"
        dominant_percent = batt_percent
        secondary_station = "Kandy"
        dominant_value = batt_pred
        secondary_value = kandy_pred
    else:
        dominant_station = "Kandy"
        dominant_percent = kandy_percent
        secondary_station = "Battaramulla"
        dominant_value = kandy_pred
        secondary_value = batt_pred
    
    # Generate explanations based on scenario
    explanation = ""
    extra_line = ""
    
    if scenario == "near_station":
        # Near one station (>80% dominance)
        explanation = f"This location is very close to {dominant_station}, so its reading primarily shapes the final value ({dominant_percent:.1f}% influence)."
        extra_line = f"The model still factors in {secondary_station} slightly to maintain a smooth, continuous gradient across the map."
    
    elif scenario == "leaning":
        # Slightly leaning (65-80% dominance)
        explanation = f"{dominant_station} has the strongest pull here, but {secondary_station} is close enough to visibly blend into the calculation."
        extra_line = f"Our ensemble model combines both stations to create a seamless transition, preventing sharp cutoffs between {dominant_value:.1f} and {secondary_value:.1f}."
    
    elif scenario == "middle":
        # In the middle (~50/50)
        explanation = f"You are right between {dominant_station} and {secondary_station}, so the model balances both of their readings almost equally."
        
        # Add extra line explaining the blend
        extra_line = f"By combining multiple spatial algorithms, it calculates a perfectly smooth transition that lands between {dominant_value:.1f} and {secondary_value:.1f}."
    
    return {
        "value": float(final_pred),
        "explanation": explanation,
        "extra_line": extra_line,
        "scenario": scenario,
        "distances": {
            "battaramulla_km": round(distance_batt, 2),
            "kandy_km": round(distance_kandy, 2)
        },
        "weights": {
            "battaramulla_percent": round(batt_percent, 1),
            "kandy_percent": round(kandy_percent, 1)
        },
        "station_values": {
            "battaramulla": round(batt_pred, 2),
            "kandy": round(kandy_pred, 2)
        },
        "dominant_station": dominant_station,
        "dominant_influence_percent": round(dominant_percent, 1),
        "target": target
    }
