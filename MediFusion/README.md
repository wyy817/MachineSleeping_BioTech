# 🧬MediFusion

MediFusion is a medical information intelligent service platform combined with AI. We are committed to bringing and serving more groups in the medical and health industry with the emerging technology of AI, including individual users, doctors and researchers of Machine Learning for medicine. 



## Features

- 🤖**AI Consultation:** The AI health consultation system is an AI-based health problem diagnosis platform. Users can input their symptoms, and the system will provide health diagnosis suggestions using preset AI models. The system is designed to provide users with convenient health consultation services, helping them understand possible diseases and recommending subsequent action plans. 
- 💾**Health Dataset Upload and Management:** Users can upload datasets to the system. Each dataset includes metadata such as name, description, tags, and upload time. Datasets can be tagged, deleted, and edited by their respective owners or admins.



## Installation and Usage

1. **Clone the project**

   ```
   `git clone https://github.com/wyy817/MachineSleeping_BioTech.git
   `cd MachineSleeping_BioTech/MediFusion`
   ```

2. **Create a Virtual Environment**

   ```
   python -m venv venv
   ```

3. **Install Dependencies**

   ```
   pip install -r requirements.txt
   ```

4. **Set Up Database**
   This project uses CSV files to store data. Make sure datasets.csv and the user data files are created and the paths are correct.

5. **Run the Project**

   ```
   flask run
   ```

   Visit http://127.0.0.1:5000 to view the application.



## Project Structure

```
MediFusion/
│
├── app/
│   ├── routes/                 # Handles various route requests
│   │   ├── admin.py            # Admin functionalities
│   │   ├── auth.py             # Login and registration
│   │   ├── dashboard.py        # Dashboard functionality
│   │   ├── doctor.py           # Doctor-related functionalities
│   │   ├── personal.py         # Personal consultations
│   │   └── warehouse.py        # Dataset management and upload
│   ├── config.py               # Configuration settings
│   ├── forms.py                # Forms handling
│   ├── llm.py                  # Large language model processing
│   ├── models.py               # Data models and database handling
│   └── utils.py                # Utility functions
├── static/                     
│   ├── css/                    # Stylesheets
│   │   └── style.css           # Main stylesheet
│   ├── images/                 # Image assets
│   │   ├── back0.png
│   │   ├── back1.jpg
│   │   ├── back2.png
│   │   ├── back3.jpg
│   │   └── back4.jpg
│   └── js/                     # JavaScript files
├── templates/
│   ├── auth/                   # Authentication templates
│   │   ├── login.html
│   │   └── register.html
│   ├── dashboard/              # Dashboard templates
│   │   └── index.html
│   ├── doctor/                 # Doctor-related templates
│   ├── personal/               # Personal consultation templates
│   │   └── consult.html
│   ├── warehouse/              # Warehouse-related templates
│   │   ├── my_data.html
│   │   ├── set_tags.html
│   │   └── upload.html
├── uploads/                    # Directory for file uploads
├── datasets.csv                # Dataset file
├── users.csv                   # User data
├── requirements.txt            # Dependency list
└── run.py                      # Entry point to run the Flask app
```



## Limitation

We developed a preliminary service system based on Python 3.12 and the Flask framework, which can be run locally by simply configuring the dependency files. In this demo system, we use mock LLM (llm.py) to simulate the real LLM due to the limitation of time and equipment. 



## 🌐Web Demo

![](E:\pic\0.png)

![1](E:\pic\1.png)

![2](E:\pic\2.png)

![3](E:\pic\3.png)

![4](E:\pic\4.png)