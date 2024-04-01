from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def load_data(data_dir):
    X = []
    y = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".wav"):
            audio_file = os.path.join(data_dir, filename)
            features = extract_features(audio_file)
            X.append(features)
            # Manually assign labels based on filename or extract labels from a metadata file
            if "tap" in filename:
                y.append("tap")
            elif "clap" in filename:
                y.append("clap")
            elif "snap" in filename:
                y.append("snap")

    return X, y


def train_model(data_dir):
    # Load data
    X, y = load_data(data_dir)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train model
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    # Evaluate model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy}")

    return clf
