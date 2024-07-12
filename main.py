from flask import Flask
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from openai import OpenAI

app = Flask(__name__)
API_KEY_OPEN_AI = ""

@app.route("/", methods=['GET'])
def build_model():
    data = pd.read_csv('Crop_Recommendation.csv')
    y = data['Crop'].values
    rows= ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']
    X = data[rows].values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    y = tf.keras.utils.to_categorical(y)

# Splitting the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creating a neural network using TensorFlow for multi-class classification
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer
        tf.keras.layers.Dense(64, activation='relu'),  # Hidden layer
        tf.keras.layers.Dense(y_train.shape[1], activation='softmax')  # Output layer with softmax activation for multi-class classification
    ])

    # Compiling the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Training the model
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
    # Evaluate the model
    # loss, accuracy = model.evaluate(X_test, y_test)

    # Print the accuracy
    # print(f'Test Accuracy: {accuracy * 100:.2f}%')
    soil_data = [69,57,43,26,73,6,177]
    new_data = np.array([soil_data])

    # Predict the class probabilities
    predicted_probabilities = model.predict(new_data)

    # Convert probabilities to class labels
    predicted_class_indices = np.argmax(predicted_probabilities, axis=1)

    # Decode the labels
    predicted_labels = label_encoder.inverse_transform(predicted_class_indices)

    client = OpenAI(api_key=API_KEY_OPEN_AI)
    test=client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"""You are a environment and agriculture expert. A farmer tested the soil and found that it had {soil_data[:3]} Nitrogen, Phosphorus and Potassium. It was recommended to him that he should grow {predicted_labels[0]} crop in this environment. Explain in short paragraph how can farmer mitigate the soil degradation and imrpove yield of the crop. Also tell which soil can be mixed with the current one in order to do so. Also note that its 2050 and soil has degraded and Rains are less frequent, but when it
rains, it storms. The drought-
hardened soil is unable to absorb the

sudden massive downpours of water,
leading to widespread flooding of
crops, and washing away of topsoil.
Increased Winter temperatures also
mean crops enter a hardened
dormancy state less frequently,
making them more vulnerable to such
damage. Keep your answers within a max limit of 150 words and a high school level vocab"""}],
        
    )
    # print(test.choices[0].message.content)
    return(f'Ideal Crop: {predicted_labels[0]}\n Model Advice: {test.choices[0].message.content}')    
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)