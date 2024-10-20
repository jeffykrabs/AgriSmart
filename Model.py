import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier






# Load the dataset
try:
   crop_data = pd.read_csv('./Crop_recommendation.csv')
except Exception as e:
   st.error(f"Error loading data: {e}")


# Extract features and labels
crop_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
X = crop_data[crop_features]
y = crop_data['label']


# Train the DecisionTreeClassifier model
model = DecisionTreeClassifier(random_state=1)
model.fit(X, y)


# Streamlit Tabs
tab1, tab2, tab3 = st.tabs(["🌱 Crop Recommendation", "📊 Data Visualizations", "📝 Input Your Data"])


# Tab 1: Crop Recommendation System
with tab1:
   st.title('🌾 Crop Recommendation System')
   st.write("""
   Welcome! This application predicts the most suitable crop based on soil and environmental conditions.
   Please enter the values below to get a crop recommendation.
   """)


   # User inputs
   st.subheader("Enter Soil and Environmental Conditions")
   N = st.number_input('Nitrogen content (N)', min_value=0.0, max_value=100.0, value=50.0)
   P = st.number_input('Phosphorus content (P)', min_value=0.0, max_value=100.0, value=50.0)
   K = st.number_input('Potassium content (K)', min_value=0.0, max_value=100.0, value=50.0)
   temperature = st.number_input('Temperature (°C)', min_value=-10.0, max_value=50.0, value=25.0)
   humidity = st.number_input('Humidity (%)', min_value=0.0, max_value=100.0, value=50.0)
   ph = st.number_input('Soil pH level', min_value=0.0, max_value=14.0, value=6.5)
   rainfall = st.number_input('Rainfall (mm)', min_value=0.0, max_value=500.0, value=100.0)


   # Prediction and result display
   if st.button('Predict Optimal Crop'):
       st.subheader("Prediction Results")


       input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=crop_features)


       try:
           prediction = model.predict(input_data)
           recommended_crop = prediction[0]
           st.success(f'🌱 The recommended crop is: **{recommended_crop}**')


       except Exception as e:
           st.error(f"Prediction error: {e}")


# Tab 2: Data Visualizations
with tab2:
   st.title('📊 Crop Data Visualizations')


   # Get unique crop labels
   crop_labels = crop_data['label'].unique()


   # Initialize checkboxes in session state
   if 'selected_crops_state' not in st.session_state:
       st.session_state.selected_crops_state = {crop: True for crop in crop_labels}


   # Functionality for 'Select All' and 'Deselect All'
   select_all = st.button("Select All")
   deselect_all = st.button("Deselect All")


   if select_all:
       st.session_state.selected_crops_state = {crop: True for crop in crop_labels}
   if deselect_all:
       st.session_state.selected_crops_state = {crop: False for crop in crop_labels}


   # Create checkboxes with columns layout (for better display)
   cols = st.columns(3)  # Display checkboxes in 3 columns
   for idx, crop in enumerate(crop_labels):
       col_idx = idx % 3  # This will distribute the crops across 3 columns
       with cols[col_idx]:
           st.session_state.selected_crops_state[crop] = st.checkbox(crop, value=st.session_state.selected_crops_state[crop])


   # Filter the dataset based on selected crops
   selected_crops = [crop for crop, is_selected in st.session_state.selected_crops_state.items() if is_selected]
  
   if selected_crops:
       filtered_data = crop_data[crop_data['label'].isin(selected_crops)]


       # Dynamic Range Selection Sliders
       st.subheader("Adjustable Graph Ranges")


       # Temperature (x-axis) Range
       min_temp, max_temp = st.slider('Temperature Range (°C)', min_value=int(filtered_data['temperature'].min()),
                                       max_value=int(filtered_data['temperature'].max()), value=(int(filtered_data['temperature'].min()), int(filtered_data['temperature'].max())))
      
       # Feature for y-axis
       feature_option = st.selectbox("Select Feature for Y-Axis", ['N', 'P', 'humidity', 'rainfall'])
       min_y, max_y = st.slider(f"{feature_option} Range", min_value=int(filtered_data[feature_option].min()),
                                 max_value=int(filtered_data[feature_option].max()), value=(int(filtered_data[feature_option].min()), int(filtered_data[feature_option].max())))


       # Nitrogen vs Temperature (or selected feature), colored by crop
       st.subheader(f"{feature_option} vs Temperature (Zoomable)")


       # Filter data based on selected range
       filtered_range_data = filtered_data[(filtered_data['temperature'] >= min_temp) & (filtered_data['temperature'] <= max_temp) &
                                           (filtered_data[feature_option] >= min_y) & (filtered_data[feature_option] <= max_y)]


       fig, ax = plt.subplots(figsize=(10, 6))
       sns.scatterplot(data=filtered_range_data, x='temperature', y=feature_option, hue='label', ax=ax, palette="Set2")
       ax.set_title(f'{feature_option} vs Temperature for Selected Crops (Range {min_temp}°C to {max_temp}°C)')
       st.pyplot(fig)


       # Reset button to display the full dataset
       if st.button("Reset to Full View"):
           min_temp = int(filtered_data['temperature'].min())
           max_temp = int(filtered_data['temperature'].max())
           min_y = int(filtered_data[feature_option].min())
           max_y = int(filtered_data[feature_option].max())


           fig, ax = plt.subplots(figsize=(10, 6))
           sns.scatterplot(data=filtered_data, x='temperature', y=feature_option, hue='label', ax=ax, palette="Set2")
           ax.set_title(f'{feature_option} vs Temperature for Selected Crops (Full Range)')
           st.pyplot(fig)


   else:
       st.warning("Please select at least one crop to display the graphs.")


# Farming practices for each crop
farming_practices = {
   'pigeonpeas': """
       - **Seed Spacing**: 15-20 cm between seeds, 60 cm between rows.
       - **Water Requirements**: Moderate, about 400-600 mm during the growing season.
       - **Soil Quality**: Well-drained, sandy or loamy soil, pH 6.0-7.5.
       - **Plot Shape**: Rectangular or square plots.
       - **Additional Tips**:
           - Pigeonpeas are drought-tolerant but need adequate moisture during flowering.
           - Apply organic manure to improve soil fertility.
           - Control weeds during the early stages of growth.
   """,
  
   'rice': """
       - **Seed Spacing**: 20-30 cm between seeds, 25 cm between rows.
       - **Water Requirements**: High, about 1000-1500 mm of water during the growing season.
       - **Soil Quality**: Clay to silty clay soil, pH 5.5-7.0.
       - **Plot Shape**: Flat plots with proper irrigation systems.
       - **Additional Tips**:
           - Maintain flooded conditions during the growing season.
           - Apply phosphorus and potassium fertilizers before planting.
           - Use integrated pest management practices to control pests like rice stem borers.
   """,
  
   'maize': """
       - **Seed Spacing**: 25-30 cm between seeds, 75 cm between rows.
       - **Water Requirements**: High, about 500-800 mm during the growing season.
       - **Soil Quality**: Well-drained, fertile soils, pH 5.8-7.0.
       - **Plot Shape**: Rectangular plots with proper irrigation.
       - **Additional Tips**:
           - Apply nitrogen fertilizer in stages to avoid leaching.
           - Ensure adequate spacing for better air circulation and sunlight penetration.
           - Control weed growth early to avoid competition for nutrients.
   """,
  
   'chickpea': """
       - **Seed Spacing**: 15-20 cm between seeds, 30 cm between rows.
       - **Water Requirements**: Moderate, around 400-500 mm of water during growing season.
       - **Soil Quality**: Well-drained, slightly acidic to neutral soil (pH 6.0-7.0).
       - **Plot Shape**: Rectangular or square plots.
       - **Additional Tips**:
           - Chickpeas are drought-tolerant, but they require good soil moisture during germination.
           - Use organic fertilizers to enhance soil fertility.
           - Control aphids and other pests using integrated pest management (IPM) techniques.
   """,
  
   'kidneybeans': """
       - **Seed Spacing**: 5-7 cm between seeds, 30-40 cm between rows.
       - **Water Requirements**: Moderate, around 450-600 mm during the growing season.
       - **Soil Quality**: Well-drained, loamy soil, pH 6.0-7.5.
       - **Plot Shape**: Rectangular or square plots.
       - **Additional Tips**:
           - Beans thrive in slightly acidic soil. Avoid waterlogging to prevent root rot.
           - Apply phosphorus and potassium fertilizers to boost yields.
           - Control weeds early to prevent competition for nutrients.
   """,
  
   'mothbeans': """
       - **Seed Spacing**: 10-15 cm between seeds, 30 cm between rows.
       - **Water Requirements**: Low to moderate, around 250-500 mm of water.
       - **Soil Quality**: Well-drained, sandy loam soil, pH 6.0-7.0.
       - **Plot Shape**: Rectangular or square plots.
       - **Additional Tips**:
           - Moth beans are drought-tolerant and require minimal irrigation.
           - Use organic or green manure to improve soil fertility.
           - Protect from pests like aphids and termites.
   """,
  
   'mungbean': """
       - **Seed Spacing**: 5-7 cm between seeds, 30 cm between rows.
       - **Water Requirements**: Moderate, around 400-600 mm during growing season.
       - **Soil Quality**: Well-drained, sandy loam or clay loam, pH 6.0-7.0.
       - **Plot Shape**: Rectangular or square plots.
       - **Additional Tips**:
           - Mungbeans grow best in warm, dry conditions.
           - Avoid over-irrigation, as it can cause root rot.
           - Apply organic fertilizers and protect from pests like beetles.
   """,
  
   'blackgram': """
       - **Seed Spacing**: 10-15 cm between seeds, 30 cm between rows.
       - **Water Requirements**: Moderate, around 400-600 mm during growing season.
       - **Soil Quality**: Well-drained, sandy loam, pH 6.0-7.5.
       - **Plot Shape**: Rectangular or square plots.
       - **Additional Tips**:
           - Blackgram requires minimal irrigation; avoid waterlogging.
           - Use organic fertilizers and control weeds early.
           - Monitor for pest infestations such as aphids.
   """,
  
   'lentil': """
       - **Seed Spacing**: 10-15 cm between seeds, 20 cm between rows.
       - **Water Requirements**: Low to moderate, around 300-450 mm of water.
       - **Soil Quality**: Well-drained, loamy soil, pH 6.0-7.0.
       - **Plot Shape**: Rectangular or square plots.
       - **Additional Tips**:
           - Lentils grow best in cool temperatures, so early planting is beneficial.
           - Avoid over-watering to prevent fungal diseases.
           - Use crop rotation to maintain soil fertility and prevent pest build-up.
   """
}




# Tab 3: Farming Practices for Crops
with tab3:
   st.title(' Crop Farming Practices')


   st.write("""
   Here you can find detailed farming practices for various crops, including seed spacing, water requirements, soil quality, plot shape, and more.
   Select a crop to learn the best practices for its cultivation.
   """)


   # Get crops available for selection in Tab 3
   crop_labels = crop_data['label'].unique()


   # Initialize session state for selected crop if not already present
   if 'selected_crop' not in st.session_state:
       st.session_state.selected_crop = crop_labels[0]  # Default to the first crop


   # Dropdown for selecting crops
   selected_crop = st.selectbox("Select a Crop to Learn Best Practices", crop_labels, index=crop_labels.tolist().index(st.session_state.selected_crop))  # Remove .tolist()


   # Store selected crop in session state to keep track of the selection
   st.session_state.selected_crop = selected_crop


   # Display farming practices for the selected crop
   if selected_crop in farming_practices:
       st.subheader(f"Farming Practices for {selected_crop}")
       st.write(farming_practices[selected_crop])
   else:
       st.warning(f"The farming practices for {selected_crop} are not available. Please select a valid crop.")

