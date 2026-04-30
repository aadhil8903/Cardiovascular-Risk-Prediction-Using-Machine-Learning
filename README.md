# Cardiovascular Risk Prediction Using Machine Learning

This project is a web-based application that predicts cardiovascular risk based on user health inputs. It uses a logistic regression model and provides results along with a simple explanation to help users understand their risk level.

---

## Features

- Takes inputs such as age, blood pressure, cholesterol, physical activity, and BMI  
- Predicts risk level: **Low, Moderate, High, Critical**  
- Provides a basic explanation based on input values  
- Deployed using Flask on AWS EC2  

---

## Tech Stack

- **Backend:** Python (Scikit-learn, Pandas, NumPy), Flask  
- **Frontend:** HTML, CSS, JavaScript  
- **Deployment:** AWS EC2  

---

## How to Run Locally

```bash
pip install -r requirements.txt
python app.py
📁 Project Structure
├── app.py              # Main application
├── model.pkl           # Trained ML model
├── templates/          # HTML files
├── static/             # CSS and JavaScript files
├── requirements.txt    # Dependencies
⚠️ Disclaimer

This is a student project and should not be considered a substitute for professional medical advice.

📜 License

This project is licensed under the MIT License.

🌐 Live Demo

https://medai-app.duckdns.org/
