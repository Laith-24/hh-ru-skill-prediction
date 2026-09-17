$env:PATH = "C:\Users\LAITH AHMAD\AppData\Local\Programs\Python\Python312;C:\Users\LAITH AHMAD\AppData\Local\Programs\Python\Python312\Scripts;" + $env:PATH
cd "C:\Users\LAITH AHMAD\Desktop\BD\hh-ru-skill-prediction"
streamlit run interface/skill_value_app.py --server.port 8502 --server.headless true
