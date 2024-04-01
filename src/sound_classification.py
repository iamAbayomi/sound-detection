def classify_sound(model, features):
    """
    Classifies a sound using a trained model.
    
    Parameters:
    - model: Trained machine learning model.
    - features: List of extracted features.
    
    Returns:
    - prediction: Predicted label for the sound.
    """
    prediction = model.predict([features])[0]
    return prediction
