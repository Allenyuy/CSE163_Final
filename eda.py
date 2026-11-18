import pandas as pd
import plotly.express as px



def main():
    pressure = pd.read_csv("PS1.txt", sep="\t", header=None)
    flow = pd.read_csv("FS1.txt", sep="\t", header=None)
    temperature = pd.read_csv("TS1.txt", sep="\t", header=None)
    profile = pd.read_csv("profile.txt", sep="\t", header=None)

    print("Pressure missing:", pressure.isnull().sum().sum())
    print("Flow missing:", flow.isnull().sum().sum())
    print("Temperature missing:", temperature.isnull().sum().sum())
    print("Profile missing:", profile.isnull().sum().sum())

    print("Pressure size:", get_size(pressure))
    print("Flow size:", get_size(flow))
    print("Temperature size:", get_size(temperature))
    print("Profile size:", get_size(profile))

    pressure_features = extract_features(pressure, "PS1")
    flow_features = extract_features(flow, "FS1")
    temperature_features = extract_features(temperature, "TS1")

    features = pressure_features.merge(
        flow_features,
        left_index=True,
        right_index=True
    )

    features = features.merge(
        temperature_features,
        left_index=True,
        right_index=True
    )

    print(features.head())
    print(features.shape)

    
    print("feature.shape:", features.shape)

    valve = pd.DataFrame({
        "valve_condition": profile[1].astype("category")
    })
    print("Description of features:", features.describe())

    features = features.merge(
        valve,
        left_index=True,
        right_index=True
    )

    print("Valve condition frequencies:")
    print(features["valve_condition"].value_counts().sort_index())

    plot_valve_distribution(features)
    plot_pressure(features)
    plot_flow(features)
    plot_temperature(features)
# get the size of the files
def get_size(data):
    """Return the number of rows and columns in the dataset."""
    return data.shape

def extract_features(data, sensor_name):
    """
    Returns the mean, standard deviation, and maximum
    of each operating cycle in a DataFrame.
    """
    features = pd.DataFrame({
        sensor_name + "_mean": data.mean(axis=1),
        sensor_name + "_std": data.std(axis=1),
        sensor_name + "_max": data.max(axis=1)
    })

    return features

def plot_valve_distribution(features):
    """
    Creates a bar chart showing the number of cycles
    for each valve condition.
    """
    counts = features["valve_condition"].value_counts().sort_index()

    fig = px.bar(
        x=counts.index.astype(str),
        y=counts.values,
        title="Distribution of Valve Conditions",
        labels={
            "x": "Valve Condition",
            "y": "Number of Operating Cycles"
        }
    )

    fig.write_image("valve_distribution.png")
    fig.show()

def plot_pressure(features):
    """
    Creates a box plot comparing average pressure
    across different valve conditions.
    """
    fig = px.box(
        features,
        x="valve_condition",
        y="PS1_mean",
        title="Mean Pressure by Valve Condition",
        labels={
            "valve_condition": "Valve Condition",
            "PS1_mean": "Mean Pressure (bar)"
        }
    )

    fig.update_xaxes(type="category")
    fig.write_image("pressure_boxplot.png")
    fig.show()


def plot_flow(features):
    """
    Creates a box plot comparing average flow rate
    across different valve conditions.
    """
    fig = px.box(
        features,
        x="valve_condition",
        y="FS1_mean",
        title="Mean Flow Rate by Valve Condition",
        labels={
            "valve_condition": "Valve Condition",
            "FS1_mean": "Mean Flow Rate (L/min)"
        }
    )

    fig.update_xaxes(type="category")
    fig.write_image("flow_boxplot.png")
    fig.show()

def plot_temperature(features):
    """
    Creates a box plot comparing average temperature
    across different valve conditions.
    """
    fig = px.box(
        features,
        x="valve_condition",
        y="TS1_mean",
        title="Mean Temperature by Valve Condition",
        labels={
            "valve_condition": "Valve Condition",
            "TS1_mean": "Mean Temperature (°C)"
        }
    )

    fig.update_xaxes(type="category")
    fig.write_image("temperature_boxplot.png")
    fig.show()

if __name__ == "__main__":
    main()
    
