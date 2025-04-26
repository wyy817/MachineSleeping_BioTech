# MediFusion

MediFusion is a medical information intelligent service platform combined with AI. We are committed to bringing and serving more groups in the medical and health industry with the emerging technology of AI, including individual users, doctors and researchers of Machine Learning for medicine. 



## 🤖 Part  A:  AI health consultation

### Introduction

The AI health consultation system is an AI-based health problem diagnosis platform. Users can input their symptoms, and the system will provide health diagnosis suggestions using preset AI models. The system is designed to provide users with convenient health consultation services, helping them understand possible diseases and recommending subsequent action plans. 

### Motivation

In our previous blended learning session, we had an exchange with a senior manager in health data science. He shared with us that the early online doctor consultation services on the market are facing challenges. When patients cannot easily access offline services from large hospitals, they tend to prefer either the online services of large hospitals or the offline services of small and medium-sized hospitals. This market phenomenon inspired us to provide preliminary advice for individual users rather than completely replacing human services.

### Implement

The module is open to all users. We developed a preliminary service system based on Python 3.12 and the Flask framework, which can be run locally by simply configuring the dependency files. In this demo system, we use mock LLM (llm.py) to simulate the real LLM due to the limitation of time and equipment. 

### Run Project

1. Clone or download the project

2. Install dependencies
   Navigate to the project root directory and run the following command to install the required Python libraries:

   ```
   pip install -r requirements.txt
   ```

3. Run the application

   In the terminal, enter the project directory and start the Flask development server with:

   ```
   flask run
   ```

4. By default, the application will launch at: http://127.0.0.1:5000/  

5. Access the application: 

   Open your browser and visit http://127.0.0.1:5000 to start using the AI Health Advisory System. 

### Web Demo

![](C:\Users\Think\Pictures\Screenshots\a1.png)

![](C:\Users\Think\Pictures\Screenshots\a2.png)

