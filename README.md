# AI-driven COVID health stats predictor
This software is developed for predicting COVID-19 severity using Convolutional Neural Networks (CNN) and Long Short-Term Memory Networks (LSTM). It assesses severity by analyzing protein sequences in FASTA format, alongside clade, gender, age, and lineage data. Additionally, it automates the installation of necessary packages.

After downloading the software (AI-driven_COVID_predictor.py) and the trained model (model_trained.h5), grant all necessary permissions using `chmod`. Once the permissions are set, you can run the software by calling it with Python 3.

### Required Data:
- Protein sequences in FASTA format
- Clade information
- Gender
- Age
- Lineage data

### How to Run:
```bash
chmod +x AI-driven_COVID_predictor.py
python3 AI-driven_COVID_predictor.py
```

The user will be prompted to enter the required data manually in the terminal.
