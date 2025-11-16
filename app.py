#Libraries
import streamlit as st 
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np 
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Basic USage of StreamLit
#title function
st.title("Assignment: Streamlit Tool Manual")
#header function
st.header("Course : Data Science Tools and Techniques")
#subheader function
st.subheader("Prepared by: Arooba Minhas (25K-8016)")
#image function
st.image("C:/Users/Supreme Traders/Desktop/Streamlit_Project/images.png", 
         caption="Streamlit Logo :sparkles:", width = 400)

#information function
st.info("This script imports the Streamlit library and uses its built-in functions to display a title, header, subheader, and an information message in a simple web interface.")
#success function
st.success("App created Successfully")
#write function
st.write("Welcome to my Streamlit app!")
#warning function
st.warning("This action is not recommended!")
#error function
st.error("Failed to load data!")

#markdown functions
st.markdown("# This is a H1 Heading")
st.markdown("## This is a H2 Heading")
st.markdown("### This is a H3 Heading")
st.markdown("#### This is a H4 Heading")
st.markdown("##### This is a H5 Heading")
st.markdown("###### This is a H6 Heading")
st.markdown("**This text is bold**")
st.markdown("_This text is italic_")
st.markdown("[This is a link](https://www.streamlit.io)")

#with symbols
st.markdown("# Streamlit Project Dashboard :computer:")  
st.markdown("## Data Upload Section :file_folder:")  
st.markdown("### Visualizations :bar_chart:")  
st.markdown("#### Insights :bulb:")  
st.markdown("**Tasks Completed** :tick:")

#text function
st.text("This is a simple text message to show the app status.")

#caption function
st.caption("Note: All columns must be numeric for analysis.")

#mathematical function
st.latex(r"E = mc^2")
st.latex(r"\frac{a}{b} = c")
st.latex(r"\sum_{i=1}^{n} i = \frac{n(n+1)}{2}")


st.title("Streamlit Widgets Example")
#Button
st.button("Click Me")
#Checkbox
st.checkbox("Check Me")
#Radio Button
st.radio("Choose one", ["Option 1", "Option 2"])
#Selectbox
st.selectbox("Pick one", ["Option 1", "Option 2", "Option 3"])
#Multiselect
st.multiselect("Select multiple", ["Option A", "Option B", "Option C"])
#Slider
st.slider("Select value", 0, 100, 50)
#Text input
st.text_input("Enter text")
#Text area
st.text_area("Enter multiple lines")
#Number input
st.number_input("Enter number", 0, 100, 10)
#Date input
st.date_input("Pick a date", datetime.today())
#Time input
st.time_input("Pick a time", datetime.now().time())
#File uploader
st.file_uploader("Upload file")
#Color picker
st.color_picker("Pick a color", "#00f900")
#Select slider
st.select_slider("Choose range", options=[1, 2, 3, 4, 5])
#Progress bar (static example at 50%)
st.progress(50)

# Sidebar
st.sidebar.title("Tool Manual") 
# Text inputs directly
st.sidebar.text_input("Email")
st.sidebar.text_input("Password", type="password")  # hides input
# Radio buttons directly
st.sidebar.radio("Select Role", ["Student", "Professional", "Expert"])
# Button directly
st.sidebar.button("Submit")


#Now displaying data using pandas
#Simple Table
data = {
    "Country": ["Pakistan", "USA", "UK", "Canada"],
    "GDP (Trillion $)": [1.47, 25.5, 3.1, 2.1]
}
st.header("Country GDP Table")
st.table(data)

df = pd.DataFrame({
    'Name': ['Arooba', 'Ali', 'Abdullah', 'Azhar', 'Bushra'],
    'Score': [85, 90, 78, 90, 80]
})
st.header("Student Scores Table")
st.dataframe(df)

#Bar Chart
st.title("Bar Chart")
data=pd.DataFrame(np.random.randn(50,2), columns=["x","y"])
st.bar_chart(data)

#Line Chart 
df = pd.DataFrame({
    'a': [1,2,3,4],
    'b': [4,3,2,1]
})
st.header("Line Chart")
st.line_chart(df)

#Matplotline plot
x = [1,2,3,4,5]
y = [10, 8, 6, 4, 2]
fig, ax = plt.subplots()
ax.plot(x, y, marker='o')
ax.set_title("Matplotlib Line Plot")
st.pyplot(fig)

#seaborn plot 
tips = sns.load_dataset("tips")
fig = plt.figure(figsize=(6,4))
sns.scatterplot(data=tips, x="total_bill", y="tip")
st.header("Seaborn Scatter Plot")
st.pyplot(fig)


#plotly plot
df = px.data.gapminder().query("year == 2007")
fig = px.scatter(
    df, x="gdpPercap", y="lifeExp", size="pop",
    color="continent", hover_name="country", log_x=True
)
st.header("Plotly Chart")
st.plotly_chart(fig)


#Advance Features
st.set_page_config(page_title="Iris Dataset App", layout="wide")
# Title Section
st.title("Iris Dataset")
st.write("This app allows you to explore the Iris dataset, visualize features, and train a simple ML model.")
# Load Dataset
@st.cache_data
def load_data():
    return sns.load_dataset("iris")

df = load_data()
st.subheader("Dataset Preview")
st.dataframe(df.head())


# Summary Statistics
st.subheader("Summary Statistics")
st.write(df.describe())

# Sidebar inputs
st.sidebar.header("Input Features")

sl = st.sidebar.slider("Sepal Length", float(df.sepal_length.min()), float(df.sepal_length.max()))
sw = st.sidebar.slider("Sepal Width", float(df.sepal_width.min()), float(df.sepal_width.max()))
pl = st.sidebar.slider("Petal Length", float(df.petal_length.min()), float(df.petal_length.max()))
pw = st.sidebar.slider("Petal Width", float(df.petal_width.min()), float(df.petal_width.max()))

# Visualizations Section
st.header("Data Visualizations")

# Scatter Plot (Plotly)
st.subheader("Scatter Plot (Plotly)")
fig1 = px.scatter(
    df,
    x="sepal_length",
    y="petal_length",
    color="species",
    size="sepal_width",
    hover_data=["petal_width"],
    title="Scatter Plot of Iris Features"
)
st.plotly_chart(fig1, use_container_width=True)

#Histogram
st.subheader("Histograms for All Features")
num_cols = df.select_dtypes(include='float').columns

for col in num_cols:
    fig2 = px.histogram(df, x=col, color="species", title=f"Histogram of {col}")
    st.plotly_chart(fig2, use_container_width=True)

#Boxplots
st.subheader("Box Plots for All Features")
for col in num_cols:
    fig3 = px.box(df, x="species", y=col, title=f"Boxplot of {col} by Species")
    st.plotly_chart(fig3, use_container_width=True)

# Pairplot
st.subheader("Seaborn Pairplot")
pairplot_fig = sns.pairplot(df, hue="species")
st.pyplot(pairplot_fig)

#Heatmap
st.subheader("Correlation Heatmap")
plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(df.iloc[:, :-1].corr(), annot=True, cmap="coolwarm")
st.pyplot(heatmap.figure)


#ML Model Training

st.header("Machine Learning Model using Logistic Regression")
# Split dataset
X = df.drop("species", axis=1)
y = df["species"]
# Encode labels
y = y.replace({"setosa": 0, "versicolor": 1, "virginica": 2})
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Prediction accuracy
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Accuracy")
st.write(f"**Accuracy:** {accuracy:.2f}")


# Prediction Interface
st.header("Predict Iris Species")
# Make prediction
input_data = scaler.transform([[sl, sw, pl, pw]])
prediction = model.predict(input_data)[0]

species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

st.subheader("Predicted Species")
st.write(f"**{species_map[prediction]}**")