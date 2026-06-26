### WaveSense AI  
### WiFi-Based Human Gesture Recognition using CNN + GRU

WaveSense AI is a deep learning-based system that recognizes human gestures using WiFi Channel State Information (CSI) signals. It uses a hybrid CNN + GRU architecture to extract spatial and temporal patterns from wireless signals and classify different human gestures.

The project also includes an interactive Streamlit dashboard for visualization, dataset exploration, and model performance analysis.

---

### Features

- 📡 WiFi CSI-based human gesture recognition  
- 🧠 CNN + GRU deep learning model  
- ⚡ Real-time prediction pipeline  
- 📊 Interactive Streamlit dashboard  
- 📈 Training accuracy & loss visualization  
- 🔍 Confusion matrix analysis  
- 📁 Dataset exploration tools  
- 🧪 Model evaluation reports  

---

## 🏗️ Project Structure
├── app.py # Main Streamlit application
├── model.py # CNN-GRU model architecture
├── dataset.py # Dataset loading & preprocessing
├── utils.py # Helper functions
├── predict.py # Inference script
├── train.py # Model training script

├── outputs/
│ ├── wavesense_best.pth
│ ├── confusion_matrix.png
│ ├── loss_curve.png
│ ├── accuracy_curve.png
│ ├── training_report.txt
│ └── label_mapping.txt

├── pages/
│ ├── 1_Dashboard.py
│ ├── 2_Visualization.py
│ ├── 3_Model_Performance.py
│ ├── 4_Dataset_Explorer.py
│ └── 5_About_Model.py

└── widar_env/ (excluded from GitHub)

---

## ⚙️ Installation

### Clone repository
```bash
git clone https://github.com/your-username/WaveSense-AI.git
cd WaveSense-AI

### CREATE VIRTUAL ENVIRONMENT
python -m venv venv

##ACTIVATION FOR WINDOWS
venv\Scripts\activate

##ACTIVATION FOR Mac/Linux
source venv/bin/activate

###Install Dependencies
pip install -r requirements.txt

###Run Project
streamlit run app.py

###Model Architecture
-CNN layers → Extract spatial features from WiFi CSI data
-GRU layers → Capture temporal dependencies
-Fully connected layers → Final classification output

###Results
-Training curves (accuracy & loss) in outputs/
-Confusion matrix for evaluation
-Model performance report included

###Important Notes
Dataset folder (BVP/) is excluded due to large size (>100MB GitHub limit)
Virtual environment (widar_env/) is not included in repository
If model file is missing, run train.py to retrain the model

###Links For Dataset
-https://cloud.tsinghua.edu.cn/d/2760bb9557ca4d09a74d/?p=%2FCSI&mode=list
-https://ieee-dataport.org/open-access/widar-30-wifi-based-activity-recognition-dataset
-https://tns.thss.tsinghua.edu.cn/widar3.0/

###Tech Stack
-Python
-PyTorch
-Streamlit
-NumPy
-Matplotlib
-Scikit-learn

###Future Improvements
-Real-time WiFi signal integration
-Transformer-based model upgrade
-Cloud deployment (Streamlit Cloud / AWS)
-Mobile-friendly dashboard

###Author

Sameedullah Khan



