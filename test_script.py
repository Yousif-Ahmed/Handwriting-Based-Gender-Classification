from helpers import *

input_file = input("Please enter file path of the tetset : ")
choosen_model = input("""
Please Enter model number to use it:
1- Logistic
2- LinearSVC
\n""")
windows_os = input("Is your OS Windows? Yes / No\n")
windows_os = True if windows_os.lower()=="yes" else False
model_path = "models/best_logistic_model_ever" if choosen_model == 1 else "models/best_svc_model_ever"
model_pipeline_testing(input_file , model_path, windows=windows_os)