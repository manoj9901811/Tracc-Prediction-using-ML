from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # ✅ Use a non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.neural_network import MLPRegressor
import warnings
import joblib
from flask import Flask, render_template, request, redirect, url_for
import sqlite3
import hashlib


# Dummy user data (replace with your actual user store)
users = {'testuser': {'password': 'testpassword'}}


warnings.filterwarnings('ignore')

app = Flask(__name__, static_url_path='')


# ----------------------------------------------
# 🔹 Data Preprocessing & Model Training
# ----------------------------------------------
data = pd.read_csv("static/Train.csv")

data['date_time'] = pd.to_datetime(data['date_time'])
data = data.sort_values(by=['date_time']).reset_index(drop=True)

# Create lag features
for n in range(1, 7):
    data[f'last_{n}_hour_traffic'] = data['traffic_volume'].shift(n)
data = data.dropna().reset_index(drop=True)

# Encode categorical and time-based features
data['is_holiday'] = data['is_holiday'].apply(lambda x: 1 if x != 'None' else 0)
data['hour'] = data['date_time'].dt.hour
data['month_day'] = data['date_time'].dt.day
data['weekday'] = data['date_time'].dt.weekday + 1
data['month'] = data['date_time'].dt.month
data['year'] = data['date_time'].dt.year

ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_weather = ohe.fit_transform(data[['weather_type', 'weather_description']])
encoded_weather_df = pd.DataFrame(encoded_weather, columns=ohe.get_feature_names_out())

data = pd.concat([data, encoded_weather_df], axis=1)

features = ['is_holiday', 'temperature', 'weekday', 'hour', 'month_day', 'year', 'month'] + list(encoded_weather_df.columns)
target = ['traffic_volume']

# Scale features
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
X = x_scaler.fit_transform(data[features])
y = y_scaler.fit_transform(data[target]).flatten()

# Train Model
regr = MLPRegressor(random_state=1, max_iter=500)
regr.fit(X, y)

# ----------------------------------------------
# 🔹 Function to Generate Gauge Chart
# ----------------------------------------------
def generate_gauge_chart(pred_output, traffic_condition):
    fig, ax = plt.subplots(figsize=(8, 2))

    traffic_levels = ["No Traffic", "Busy", "Heavy", "Worst"]
    colors = ["green", "yellow", "orange", "red"]
    ranges = [0, 1000, 3000, 5500, 7000]

    for i in range(len(traffic_levels)):
        ax.barh(0, width=ranges[i + 1] - ranges[i], left=ranges[i], color=colors[i], height=0.5)

    ax.scatter(pred_output, 0, color="black", s=200, label="Predicted Traffic", marker="o", edgecolors="white", zorder=3)

    ax.set_xlim(0, ranges[-1])
    ax.set_xticks([(ranges[i] + ranges[i+1]) / 2 for i in range(len(traffic_levels))])
    ax.set_xticklabels(traffic_levels, fontsize=12, fontweight="bold")

    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.set_title(f"Predicted Traffic Condition: {traffic_condition}", fontsize=16, fontweight="bold")

    graph_path = os.path.join("static", "traffic_analysis.png")
    plt.savefig(graph_path, dpi=300, bbox_inches="tight")
    plt.close()
    return graph_path

# ----------------------------------------------
# 🔹 Function to Generate Comparison Graphs
# ----------------------------------------------
def generate_comparison_graphs(hour, temperature, pred_output):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Bar Chart: Traffic by Hour
    sns.barplot(x=data["hour"], y=data["traffic_volume"], ax=axes[0, 0], color="blue").set_title("Traffic by Hour", fontsize=14)
    axes[0, 0].scatter(hour, pred_output, color="red", s=100, label="Predicted Traffic")
    axes[0, 0].legend()

    # Scatter Plot: Traffic vs Temperature
    sns.scatterplot(x=data["temperature"], y=data["traffic_volume"], ax=axes[1, 0], color="purple").set_title("Traffic vs Temperature", fontsize=14)
    axes[1, 0].scatter(temperature, pred_output, color="red", s=100, label="Predicted Traffic")
    axes[1, 0].legend()

    # Pie Chart: Traffic by Weather Type
    weather_counts = data.groupby("weather_type")["traffic_volume"].mean()
    weather_counts["Predicted"] = pred_output
    axes[1, 1].pie(weather_counts, labels=weather_counts.index, autopct="%1.1f%%", colors=sns.color_palette("pastel"))
    axes[1, 1].set_title("Traffic by Weather Type (Including Prediction)", fontsize=14)

    graph_path = os.path.join("static", "traffic_comparison.png")
    plt.savefig(graph_path, dpi=300, bbox_inches="tight")
    plt.close()
    return graph_path

# ----------------------------------------------
# 🔹 Flask Routes
# ----------------------------------------------
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predicts')
def predict_page():
    return render_template('predicts.html')

@app.route('/')
def first():
    return render_template('first.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/logout')
def logout():
    return render_template('first.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Function to register a new user
def register_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password_hash))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False  # Username already exists

# Function to check login credentials
def check_login(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()

    password_hash = hashlib.sha256(password.encode()).hexdigest()
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password_hash))
    user = c.fetchone()

    conn.close()
    return user

@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            return render_template("register.html", message="Passwords do not match.")

        if register_user(username, password):
            return redirect(url_for('login'))
        else:
            return render_template("register.html", message="Username already exists.")

    return render_template("register.html")

@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']

        user = check_login(username, password)
        if user:
            return redirect(url_for('home', user=username))
        else:
            return render_template("login.html", message="Invalid credentials. Please try again.")

    return render_template("login.html")




@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract user inputs
        is_holiday = 1 if request.form['isholiday'].lower() == 'yes' else 0
        temperature = int(request.form['temperature'])
        date_str = request.form['date']
        hour = int(request.form['time'][:2])
        month_day = int(date_str[8:])
        year = int(date_str[:4])
        month = int(date_str[5:7])
        weekday = datetime.strptime(date_str, "%Y-%m-%d").weekday() + 1

        weather_type = request.form.get('x0')
        weather_description = request.form.get('x1')
        encoded_weather = ohe.transform([[weather_type, weather_description]])
        encoded_weather_df = pd.DataFrame(encoded_weather, columns=ohe.get_feature_names_out())

        final_features = [is_holiday, temperature, weekday, hour, month_day, year, month] + list(encoded_weather_df.iloc[0])
        final_scaled = x_scaler.transform([final_features])

        # Predict traffic volume
        pred_scaled = regr.predict(final_scaled)
        pred_output = y_scaler.inverse_transform([[pred_scaled[0]]])[0][0]

        # Determine traffic condition
        if pred_output <= 1000:
            traffic_condition = "No Traffic"
        elif 1000 < pred_output <= 3000:
            traffic_condition = "Busy or Normal Traffic"
        elif 3000 < pred_output <= 5500:
            traffic_condition = "Heavy Traffic"
        else:
            traffic_condition = "Worst Case"
                       

        # Generate updated graphs
        gauge_graph = generate_gauge_chart(pred_output, traffic_condition)
        comparison_graph = generate_comparison_graphs(hour, temperature, pred_output)
        
        # ✅ Generate Traffic Analysis Graphs with Different Colors
        sns.set(style="darkgrid")
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        sns.barplot(x=data["hour"], y=data["traffic_volume"], ax=axes[0, 0], palette="coolwarm").set_title("Traffic by Hour", fontsize=14)
        sns.barplot(x=data["weekday"], y=data["traffic_volume"], ax=axes[0, 1], palette="coolwarm").set_title("Traffic by Weekday", fontsize=14)
        sns.barplot(x=data["month"], y=data["traffic_volume"], ax=axes[0, 2], palette="coolwarm").set_title("Traffic by Month", fontsize=14)
        sns.barplot(x=data["temperature"], y=data["traffic_volume"], ax=axes[1, 0], palette="coolwarm").set_title("Traffic vs Temperature", fontsize=14)
        sns.barplot(x=data["weather_type"], y=data["traffic_volume"], ax=axes[1, 1], palette="coolwarm").set_title("Traffic by Weather Type", fontsize=14)

        axes[1, 2].axis("off")  # Hide empty subplot
        plt.tight_layout()

        # ✅ Save Analysis Graph Separately
        analysis_graph_path = os.path.join("static", "traffic_analysis1.png")
        plt.savefig(analysis_graph_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        
      # Load dataset
        df = pd.read_csv("accident.csv").dropna()

        # Convert column names to lowercase for consistency
        df.columns = df.columns.str.lower()

        # Convert accident severity to numeric if necessary
        severity_mapping = {"Low": 1, "Moderate": 2, "High": 3}  # Adjust as needed
        df["accident_severity"] = df["accident_severity"].map(severity_mapping).fillna(0).astype(int)

        # Scatter Plot: Speed Limit vs. Accident Severity
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x="speed_limit", y="accident_severity", hue="traffic_density", size="traffic_density", sizes=(20, 200), data=df, palette="coolwarm", alpha=0.7)

        # Titles and Labels
        plt.title("Scatter Plot: Speed Limit vs. Accident Severity", fontsize=14, color="darkred")
        plt.xlabel("Speed Limit (km/h)")
        plt.ylabel("Accident Severity")
        plt.legend(title="Traffic Density", loc="upper left")

        # Save the scatter plot
        img_path = os.path.join("static", "scatter_accident_severity.png")
        plt.savefig(img_path)
        plt.show()  # Show graph for debugging
        plt.close()



        return render_template('output.html', Prediction=traffic_condition, gauge_graph=gauge_graph, comparison_graph=comparison_graph)

    except Exception as e:
        return f"Error: {str(e)}"
    
@app.route('/route', methods=['POST'])
def route():
    try:
        # Extract source and destination
        source = request.form['source']
        destination = request.form['destination']

        # Render the output template with route
        return render_template('output.html', Prediction="Heavy Traffic", source=source, destination=destination, route=True)

    except Exception as e:
        return f"Error: {str(e)}"
    
    # Load the trained model and preprocessing objects
model = joblib.load("accident_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

import traceback

@app.route("/accident", methods=["GET", "POST"])
def predict_accident():
    if request.method == "POST":
        try:
            # Debugging: Print received form data
            print("Received form data:", request.form)

            # Get user input and ensure proper type conversion
            data = {
                "Weather": request.form.get("weather", ""),
                "Road_Type": request.form.get("road_type", ""),
                "Time_of_Day": request.form.get("time_of_day", ""),
                "Traffic_Density": int(request.form.get("traffic_density", "0")),
                "Speed_Limit": float(request.form.get("speed_limit", "0")),
                "Number_of_Vehicles": int(request.form.get("number_of_vehicles", "0")),
                "Driver_Alcohol": int(request.form.get("driver_alcohol", "0")),
                "Accident_Severity": request.form.get("accident_severity", "Low"),
                "Road_Condition": request.form.get("road_condition", ""),
                "Vehicle_Type": request.form.get("vehicle_type", ""),
                "Driver_Age": int(request.form.get("driver_age", "0")),
                "Driver_Experience": int(request.form.get("driver_experience", "0")),
                "Road_Light_Condition": request.form.get("road_light_condition", ""),
            }

            # Encode categorical variables safely
            for col in label_encoders:
                if col in data:
                    if data[col] in label_encoders[col].classes_:
                        data[col] = label_encoders[col].transform([data[col]])[0]
                    else:
                        # If unseen category, assign default value (0 or "Unknown" if defined in training)
                        data[col] = label_encoders[col].transform(["Unknown"])[0] if "Unknown" in label_encoders[col].classes_ else 0

            # Convert data into NumPy array and scale it
            features = np.array([[data[col] for col in data]])
            features_scaled = scaler.transform(features)

            # Debugging: Print scaled features
            print("Scaled features:", features_scaled)

            # Make the prediction
            prediction = model.predict(features_scaled)[0]
            result = "High" if prediction == 1 else "Low"

            # Generate precautionary measures
            precautions = []

            if result == "High":
                precautions.append("⚠️ High accident risk detected! Please take extra precautions.")
            if data["Speed_Limit"] > 80:
                precautions.append("Reduce your speed to stay within the safe limit.")
            if data["Traffic_Density"] > 70:
                precautions.append("High traffic detected. Maintain a safe distance and avoid sudden braking.")
            if data["Driver_Alcohol"] == 1:
                precautions.append("Alcohol detected in the driver’s condition. DO NOT DRIVE under the influence!")
            if data["Road_Condition"] in ["Wet", "Icy"]:
                precautions.append("Slippery roads ahead. Drive slowly and avoid sudden turns.")
            if data["Road_Light_Condition"] in ["Poor", "Dark"]:
                precautions.append("Low visibility detected. Use headlights and be extra cautious.")
            if data["Number_of_Vehicles"] > 10:
                precautions.append("Heavy vehicle presence. Stay alert and follow lane discipline.")

            if not precautions:
                precautions.append("Drive safely and follow all traffic rules.")
            
            # Load dataset and generate analysis graphs
            df = pd.read_csv("accident.csv").dropna()

            # Set Seaborn style
            sns.set(style="whitegrid")

            # Create a figure with multiple subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))

            # 1️⃣ Traffic by Time of Day
            if 'Time_of_Day' in df.columns:
                sns.barplot(x="Time_of_Day", y="Traffic_Density", data=df, ax=axes[0, 0], color="gray")
                axes[0, 0].set_title("Traffic by Time of Day")

            # 2️⃣ Traffic by Road Type
            if 'Road_Type' in df.columns:
                sns.barplot(x="Road_Type", y="Traffic_Density", data=df, ax=axes[0, 1], palette="coolwarm")
                axes[0, 1].set_title("Traffic by Road Type")

            # 3️⃣ Traffic by Weather
            if 'Weather' in df.columns:
                sns.barplot(x="Weather", y="Traffic_Density", data=df, ax=axes[0, 2], palette="coolwarm")
                axes[0, 2].set_title("Traffic by Weather")
                axes[0, 2].tick_params(axis="x", rotation=45)

            # 4️⃣ Traffic by Speed Limit
            if 'Speed_Limit' in df.columns:
                sns.barplot(x="Speed_Limit", y="Traffic_Density", data=df, ax=axes[1, 0], palette="coolwarm")
                axes[1, 0].set_title("Traffic by Speed Limit")

            # 5️⃣ Number of Vehicles vs Road Condition
            if 'Road_Condition' in df.columns:
                sns.barplot(x="Road_Condition", y="Number_of_Vehicles", data=df, ax=axes[1, 1], palette="coolwarm")
                axes[1, 1].set_title("Number of Vehicles by Road Condition")
                axes[1, 1].tick_params(axis="x", rotation=45)

            # Remove the last empty subplot
            fig.delaxes(axes[1, 2])

            # Save the graph to the static folder
            img_path = os.path.join("static", "traffic_analysis.png")
            plt.tight_layout()
            plt.savefig(img_path)
            plt.close()
            
            df = pd.read_csv("accident.csv").dropna()

            # Filter data based on prediction result
            if result == "High":
                filtered_df = df[df["Accident_Severity"] == "High"]
            else:
                filtered_df = df[df["Accident_Severity"] == "Low"]

            sns.set(style="whitegrid")
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # 1️⃣ Accidents by Time of Day (Dynamic Highlighting)
            sns.barplot(x="Time_of_Day", y="Traffic_Density", data=filtered_df, ax=axes[0, 0], color="blue")
            axes[0, 0].set_title(f"Accidents by Time of Day ({result} Risk)")

            # Highlight the predicted time of day
            axes[0, 0].scatter(data["Time_of_Day"], data["Traffic_Density"], color="red", s=100, label="Predicted")
            axes[0, 0].legend()

            # 2️⃣ Accidents vs Speed Limit (Scatterplot)
            sns.scatterplot(x="Speed_Limit", y="Traffic_Density", data=filtered_df, ax=axes[0, 1], color="red")
            axes[0, 1].set_title(f"Accidents vs Speed Limit ({result} Risk)")

            # Highlight the predicted speed limit
            axes[0, 1].scatter(data["Speed_Limit"], data["Traffic_Density"], color="black", s=150, edgecolors="white", label="Predicted Point")
            axes[0, 1].legend()

            # 3️⃣ Pie Chart - Accident Severity Distribution
            filtered_df["Accident_Severity"].value_counts().plot.pie(autopct="%1.1f%%", ax=axes[1, 0], cmap="Pastel1")
            axes[1, 0].set_title(f"Accident Severity Distribution ({result} Risk)")

            # 4️⃣ Bar Chart - Accidents by Weather Type
            sns.barplot(x="Weather", y="Traffic_Density", data=filtered_df, ax=axes[1, 1], palette="coolwarm")
            axes[1, 1].set_title(f"Accidents by Weather Type ({result} Risk)")
            axes[1, 1].tick_params(axis="x", rotation=45)

            # Save the dynamically generated graph
            img_path = os.path.join("static", "accident_analysis.png")
            plt.tight_layout()
            plt.savefig(img_path)
            plt.close()

            # Load dataset
            df = pd.read_csv("accident.csv").dropna()

            # Convert column names to lowercase for consistency
            df.columns = df.columns.str.lower()

            # Ensure necessary columns exist
            required_columns = ["time_of_day", "traffic_density", "speed_limit", "number_of_vehicles", 
                                "accident_severity", "weather", "road_type"]
            missing_cols = [col for col in required_columns if col not in df.columns]

            if missing_cols:
                print(f"❌ Missing Columns: {missing_cols}")
            else:
                print("✅ All required columns are present.")

            # Set style
            sns.set(style="whitegrid")
            fig, axes = plt.subplots(4, 1, figsize=(12, 18))

            # Accident trend by Time of Day
            sns.lineplot(x="time_of_day", y="number_of_vehicles", data=df, ax=axes[0], color="red", marker="o", ci=90)
            axes[0].set_title("Accident Trend by Time of Day", fontsize=14, color="red")

            # Accident trend by Speed Limit
            sns.lineplot(x="speed_limit", y="number_of_vehicles", data=df, ax=axes[1], color="green", marker="o", ci=90)
            axes[1].set_title("Accident Trend by Speed Limit", fontsize=14, color="green")

            # Accident trend by Weather
            sns.barplot(x="weather", y="number_of_vehicles", data=df, ax=axes[2], palette="coolwarm")
            axes[2].set_title("Accident Trend by Weather", fontsize=14, color="blue")
            axes[2].tick_params(axis="x", rotation=45)

            # Accident trend by Road Type
            sns.barplot(x="road_type", y="number_of_vehicles", data=df, ax=axes[3], palette="magma")
            axes[3].set_title("Accident Trend by Road Type", fontsize=14, color="purple")
            axes[3].tick_params(axis="x", rotation=45)

            # Save graph
            img_path = os.path.join("static", "accident_trend_analysis.png")
            plt.tight_layout()
            plt.savefig(img_path)
            plt.close()

            print(f"✅ Graph saved at {img_path}")
            
            # Load Dataset
            df = pd.read_csv("accident.csv").dropna()

            # **Filter dataset for similar cases based on prediction**
            filtered_df = df[
                (df["Speed_Limit"] == data["Speed_Limit"]) &
                (df["Traffic_Density"] == data["Traffic_Density"]) &
                (df["Road_Condition"] == data["Road_Condition"])
            ]

            # If no matching data, use the whole dataset
            if filtered_df.empty:
                filtered_df = df

            # Set up figure for multiple pie charts
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # **Pie Chart 1: Accident Severity**
            filtered_df["Accident_Severity"].value_counts().plot.pie(
                autopct="%1.1f%%", ax=axes[0], cmap="Pastel1"
            )
            axes[0].set_title("Accident Severity (Filtered)")

            # **Pie Chart 2: Weather Conditions**
            filtered_df["Weather"].value_counts().plot.pie(
                autopct="%1.1f%%", ax=axes[1], cmap="coolwarm"
            )
            axes[1].set_title("Weather Conditions (Filtered)")

            # **Pie Chart 3: Road Condition**
            filtered_df["Road_Condition"].value_counts().plot.pie(
                autopct="%1.1f%%", ax=axes[2], cmap="Set3"
            )
            axes[2].set_title("Road Condition (Filtered)")

            # Save Graph
            pie_chart_path = os.path.join("static", "filtered_pie_charts.png")
            plt.tight_layout()
            plt.savefig(pie_chart_path)
            plt.close()

            return render_template(
                "result.html",
                prediction=result,
                precautions=precautions,
                graph_path=img_path  # Fixed graph path reference
            )

        except Exception as e:
            print("Error occurred:", traceback.format_exc())  
            return render_template("result.html", prediction="Error", precautions=["Data processing issue. Please check input values."])

    return render_template("accident_form.html")


if __name__ == '__main__':
    app.run(debug=True)
