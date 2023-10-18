import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv("cars_data.csv")
df = df.dropna()
df["MSRP"] = df["MSRP"].str.replace("$", "")
df["MSRP"] = df["MSRP"].str.replace(",", "")
df["MSRP"] = df["MSRP"].astype("int")
df_new = pd.get_dummies(df, columns= ['Make', 'Model', 'Type', 'Origin', 'DriveTrain'])
X = df_new.drop("MSRP", axis=1)
y = df_new["MSRP"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                   random_state=2)

# Train the model
model = LinearRegression()
model.fit(X, y)

# Train the model using Random Forest
random_model = RandomForestRegressor()
random_model.fit(X_train, y_train)
random_model.score(X_test, y_test)

# Train the model using Decision Tree
dec_model = DecisionTreeRegressor()
dec_model.fit(X_train, y_train)
dec_model.score(X_test, y_test)

# Define the Streamlit app
st.title("Driviction")
st.title("Car Price Predictor")
st.write("Enter the details of the car to get its estimated price")
# Write the model accuracy in the sidebar
st.sidebar.image("https://i.pinimg.com/originals/e7/81/08/e78108cc9125e2944ece1daf56231ee9.jpg", width=200)
st.sidebar.title("Live Model Accuracy Tracking")
st.sidebar.markdown("---")
st.sidebar.header("1. Linear Regression")
st.sidebar.write("Linear Regression Model Accuracy: ", round(model.score(X, y), 2))
st.sidebar.header("2. Random Forest Model")
st.sidebar.write("Random Forest Model Accuracy: ", round(random_model.score(X_test, y_test), 2))
st.sidebar.header("3. Decision Tree Model")
st.sidebar.write("Decision Tree Model Accuracy: ", round(dec_model.score(X_test, y_test), 2))
background_images = {
    "Acura": "https://cdn.wallpapersafari.com/20/12/A2OCUo.jpg",
    "Audi": "https://c4.wallpaperflare.com/wallpaper/777/369/220/audi-s5-audi-car-blue-cars-wallpaper-preview.jpg",
    "BMW": "https://wallpapers.com/images/featured/d942a3zxd8i3uqc8.jpg",
    "Volvo":"https://i.pinimg.com/originals/ea/18/47/ea18471ae332ef238b2d4a29f40d2e70.webp",
    "Buick":"https://i.pinimg.com/originals/73/0c/42/730c42f8395d09aa6ade41f9056fb38c.webp",
    "Cadillac": "https://i.pinimg.com/originals/fa/73/8b/fa738b225b147382012ef515aa9e4851.webp",
    "Toyota":"https://i.pinimg.com/originals/a6/0b/88/a60b884d7df167fa94da88e1427d5964.webp",
    "Suzuki":"https://i.pinimg.com/originals/4b/40/36/4b40367a4d5ff6e9a381a28311acd1b1.webp",
    "Volkswagen":"https://i.pinimg.com/originals/e7/b6/f5/e7b6f50583325f9d3d45f6a2ce44af94.webp",
    "Jaguar":"https://picstatio.com/large/jhv8bn/Jaguar-F-Type-sports-car-wallpaper.jpg",
    "Jeep":"https://images2.alphacoders.com/282/282022.jpg",
    "Kia":"https://www.hdcarwallpapers.com/download/2023_kia_sportage_sx_4k-3840x2160.jpg",
    "Chervolet":"https://images.pexels.com/phot"
                "os/1136566/pexels-photo-1136566.jpeg?cs=srgb&dl=pexels-tim-mossholder-1136566.jpg&fm=jpg",
    "Chrystel":"https://wallpaperaccess.com/full/1423884.jpg",
    "Dodge":"https://wallpaperaccess.com/full/2573031.jpg",
    "Ford":"https://cdn.motor1.com/images/mgl/6ZpqJk/s1/ford-mustang-electric-by-charge-cars.jpg",
    "GMC":"https://wallpaperaccess.com/full/3198306.jpg",
    "Honda":"https://www.hdcarwallpapers.com/thumbs/2022/honda_civic_type_r_2022_4k_8k_4-t2.jpg",
    "Hummer":"https://4kwallpapers.com/images/wallpapers/gmc-hummer-ev-electric-suv-2024-5k-4480x2520-5036.jpg",
    "Hyundai":"https://images.pexels.com/photos/9609144/pexels-photo-9609144.jpeg?cs=srgb&dl=pexels-aykut-ercan-9609144.jpg&fm=jpg",
    "Infiniti":"https://www.hdnicewallpapers.com/Walls/Big/Other%20Cars/Infiniti_Project_Black_S_4K_Car.jpg",
    "Isuzu":"https://images8.alphacoders.com/122/1227036.jpg",
    "Land Rover":"https://www.hdcarwallpapers.com/walls/land_rover_defender_110_country_pack_first_edition_2020_4k_2-HD.jpg",
    "Lexus":"https://images8.alphacoders.com/440/440283.jpg",
    "Lincoln":"https://www.hdcarwallpapers.com/download/2022_lincoln_navigator_black_label_5k-5120x2880.jpg",
    "MINI":"https://images.pexels.com/photos/2127037/pexels-photo-2127037.jpeg?cs=srgb&dl=pexels-maria-geller-2127037.jpg&fm=jpg",
    "Mazda":"https://wallpapersmug.com/download/3840x2160/0499a2/cars-Mazda-MX-5-4k.jpg",
    "Mercedes-Benz":"https://wallpaperaccess.com/full/1489218.jpg",
    "Mercury":"https://images.alphacoders.com/447/447569.jpg",
    "Mitsubishi":"https://wallpapercrafter.com/desktop/12994-mitsubishi-headlight-front-view-bumper-4k.jpg",
    "Nissan":"https://images.hdqwalls.com/wallpapers/nissan-gt-r-nismo-rear-4k-13.jpg",
    "Oldsmobile":"https://wallpapercave.com/wp/wp3126006.jpg",
    "Pontiac":"https://images6.alphacoders.com/322/322730.jpg",
    "Porche":"https://images5.alphacoders.com/115/1155095.jpg",
    "Saab":"https://images.wallpaperscraft.com/image/single/saab_92x_saab_car_149269_3840x2160.jpg",
    "Saturn":"https://www.carpixel.net/w/42478d149bfb89e1c0cc67a24de225b5/saturn-sky-car-wallpaper-345.jpg",
    "Scion":"https://i.pinimg.com/originals/0e/ac/43/0eac43df1cad26520cc379c7db603bca.webp",
    "Subaru":"https://wallpaperaccess.com/full/2739572.jpg",
    "Suzuki":"https://i.pinimg.com/originals/4b/40/36/4b40367a4d5ff6e9a381a28311acd1b1.webp",
    "":"",

    # add more mappings for other makes and their corresponding background images
}

# Get the selected make from the dropdown
make = st.selectbox("Make", df["Make"].unique())

# Set the background image based on the selected make
if make in background_images:
    bg_image_url = background_images[make]
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url('{bg_image_url}');
            background-attachment: fixed;
            background-size: cover
        }}
        </style>
    """, unsafe_allow_html=True)
# def add_bg_from_url():
#     st.markdown(
#          f"""
#          <style>
#          .stApp {{
#              background-image: url("https://i.pinimg.com/originals/4a/88/0b/4a880b7a9fb66c409203115abb72dd78.jpg");
#              background-attachment: fixed;
#              background-size: cover
#          }}
#          </style>
#          """,
#          unsafe_allow_html=True
#      )
#
# add_bg_from_url()
#
# def add_sbg_from_url():
#     st.markdown(
#          f"""
#          <style>
#          .e1fqkh3o3 {{
#              background-image: url("https://images.unsplash.com/photo-1584968153986-3f5fe523b044?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=987&q=80");
#              background-attachment: fixed;
#              background-size: cover
#          }}
#          </style>
#          """,
#          unsafe_allow_html=True
#      )
#
# #add_sbg_from_url()


#make = st.selectbox("Make", df["Make"].unique())
model_name = st.selectbox("Model", df[df["Make"]==make]["Model"].unique())
car_type = st.selectbox("Type", df[(df["Make"]==make) & (df["Model"]==model_name)]["Type"].unique())
origin = st.selectbox("Origin", df[(df["Make"]==make) & (df["Model"]==model_name) & (df["Type"]==car_type)]["Origin"].unique())
drive_train = st.selectbox("DriveTrain", df[(df["Make"]==make) & (df["Model"]==model_name) & (df["Type"]==car_type) & (df["Origin"]==origin)]["DriveTrain"].unique())
engine_size = st.slider("Engine Size", min_value=0.0, max_value=10.0, step=0.1)
cylinders = st.slider("Cylinders", min_value=2, max_value=12, step=1)
horsepower = st.slider("Horsepower", min_value=50, max_value=1000, step=10)
mpg_city = st.slider("MPG (City)", min_value=1, max_value=50, step=1)
mpg_highway = st.slider("MPG (Highway)", min_value=1, max_value=50, step=1)
weight = st.slider("Weight (lbs)", min_value=1000, max_value=8000, step=100)
wheelbase = st.slider("Wheelbase (inches)", min_value=50, max_value=200, step=1)
length = st.slider("Length (inches)", min_value=100, max_value=300, step=1)

# Make prediction
car = np.zeros(len(X.columns))
car[X.columns.get_loc("EngineSize")] = engine_size
car[X.columns.get_loc("Cylinders")] = cylinders
car[X.columns.get_loc("Horsepower")] = horsepower
car[X.columns.get_loc("MPG_City")] = mpg_city
car[X.columns.get_loc("MPG_Highway")] = mpg_highway
car[X.columns.get_loc("Weight")] = weight
car[X.columns.get_loc("Wheelbase")] = wheelbase
car[X.columns.get_loc("Length")] = length

make_col = "Make_" + make
model_col = "Model_" + model_name
type_col = "Type_" + car_type
origin_col = "Origin_" + origin
drive_train_col = "DriveTrain_" + drive_train

if make_col in X.columns:
    car[X.columns.get_loc(make_col)] = 1
if model_col in X.columns:
    car[X.columns.get_loc(model_col)] = 1
if type_col in X.columns:
    car[X.columns.get_loc(type_col)] = 1
if origin_col in X.columns:
    car[X.columns.get_loc(origin_col)] = 1
if drive_train_col in X.columns:
    car[X.columns.get_loc(drive_train_col)] = 1

#select the model to use for prediction
select_model = st.selectbox("Select the model to use for prediction", ["Linear Regression", "Random Forest", "Decision Tree"])
#if the selected model is Linear Regression
if select_model == "Linear Regression":
    prediction = model.predict([car])[0]
    #else if the selected model is Random Forest
elif select_model == "Random Forest":
    prediction = random_model.predict([car])[0]
    #else if the selected model is Decision Tree
elif select_model == "Decision Tree":
    prediction = dec_model.predict([car])[0]


#write the estimated price in a bold font and in green color
st.markdown("<h1 style='text-align: center; color: green;'>The estimated price of the car is: $"+str(round(prediction, 2))+"</h1>", unsafe_allow_html=True)