# Personalized Attire Recommendation System

An AI-powered system that provides personalized outfit recommendations based on user input and facial color analysis. The system uses machine learning and computer vision techniques to deliver tailored suggestions for users.

---

## 📋 Features
- **Facial Color Analysis**:
  - Extracts average colors from facial regions (e.g., forehead, nose, cheeks, lips) using dlib's facial landmark detection.
  - Generates a personalized color palette and determines the dominant seasonal color category (Spring, Summer, Autumn, Winter).

- **Body Type and Gender Selection**:
  - Interactive dropdown menu for selecting body type and gender.

- **Outfit Recommendations**:
  - Matches user inputs and dominant seasonal palette with a pre-defined outfit database (`data.csv`).
  - Provides detailed recommendations for tops, bottoms, accessories, and more.

- **Interactive Output**:
  - Displays facial color palettes and outputs recommendations as a downloadable HTML file.

---

## ⚙️ System Requirements
### Software:
- **Python 3.x**
- Libraries:
  - `cv2` (OpenCV)
  - `dlib`
  - `matplotlib`
  - `numpy`
  - `pandas`
  - `PIL` (Pillow)
  - `google.colab` (for interactive use)

### Data:
- **Pre-trained Facial Landmark Model**: `shape_predictor_68_face_landmarks.dat`
  - Automatically downloaded during execution.
- **Outfit Database**: `data.csv`
  - Columns must include:
    - `Body Type`, `Gender`, `Color Palette`, `Top`, `Bottom`, `Accessory`, `Shoe`, `Occasion`, `Weather`, `Fabric Type`.

---

## 🚀 How to Run the Project
### 1. Clone the Repository
```bash
git clone https://github.com/your-username/attire-recommendation-system.git
cd attire-recommendation-system
```

### 2. Upload `data.csv` to Your Environment
Ensure `data.csv` contains the outfit details required for recommendations.

### 3. Run in Google Colab
- Open the project in [Google Colab](https://colab.research.google.com/).
- Upload the necessary `data.csv` file.
- Execute each cell to analyze the image and generate recommendations.

### 4. Upload a Facial Image
When prompted, upload a clear image of the face.

### 5. Select Body Type and Gender
Choose from the dropdown menu for personalized recommendations.

### 6. View Recommendations
- Download the generated `outfit_recommendations.html` file for detailed suggestions.

---

## 📂 File Structure
```
├── README.md                # Project Documentation
├── shape_predictor_68_face_landmarks.dat.bz2 # Facial landmark model (downloaded during execution)
├── data.csv                 # Outfit database
├── main.py                  # Main script for execution
```

---

## 🧠 Key Functions
### 1. Facial Color Extraction
```python
def get_average_color(image, region):
    """Calculates the average color within a given region."""
```

### 2. Dominant Color Season Detection
```python
def get_dominant_color_season(colors):
    """Determines the dominant color season (Spring, Summer, Autumn, Winter)."""
```

### 3. Outfit Recommendations
```python
def recommend_outfits(user_info, dominant_season):
    """Filters the outfit database to recommend personalized outfits."""
```

---

## 🎨 Output
1. **Facial Color Palette**:
   - Displays a grid of extracted colors.
   - Identifies the dominant seasonal palette.

2. **Outfit Recommendations**:
   - Outputs recommendations as an HTML file (`outfit_recommendations.html`) with details such as:
     - Top, Bottom, Accessory, Shoe
     - Occasion, Weather, Fabric Type

---

## 🔧 Limitations
- Low-resolution images may reduce the accuracy of facial analysis.
- Recommendations depend heavily on the quality and completeness of `data.csv`.
- Performance may be slower on systems without GPU acceleration.

---

## 🛠️ Future Enhancements
- Integration with real-time e-commerce APIs for product suggestions.
- Enhanced facial analysis using deep learning.
- Support for a broader range of body types and personal preferences.

---

## 🤝 Contributing
Contributions are welcome! Please fork the repository and submit a pull request with detailed descriptions of your changes.

---

## 📝 License
This project is licensed under the [MIT License](LICENSE).
