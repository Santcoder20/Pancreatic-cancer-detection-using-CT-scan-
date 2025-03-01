Pancreatic Cancer Detection Project - Instructions


1. Setup Files:
   - Download Python 3-9 or 10.
   - Download all the files in the `Setup` folder.
   - Install the `NBIA Data Retriever-4.4` software.

2. Dataset Preparation:
   - Use the `NBIA Data Retriever-4.4` to download the dataset by clicking on `Pancreas-CT-20200910.tcia`.
   - Remove the sample provided within the `Project\pancreatic_cancer_data` directory.
   - Place the downloaded dataset files into the `pancreatic_cancer_data` folder within the `Project` directory.

3. Training and Testing:
   - Navigate to the `Project` folder.
   - Open the `Testing.py` file and go to line 10.
   - Replace `PASTE THE DIRECTORY HERE` with the directory path of the testing dataset.
   - Train and test the model using the dataset stored in the `data` folder.

4. Model Usage:
   - Go to the `Project Website` folder.
   - Run the `app` file to start the web application.
   - Upload a CT scan image to check for pancreatic cancer.

5. Additional Resources:
   - PPT: Contains the PowerPoint presentation for the project.
   - Report: Includes the final report documentation for the project.
   - Base Paper: Contains reference papers used during the project development.

6. Important Folders:
   - `Project`: For training and testing.
   - `Project Website`: To run the web app for predictions.
   - `PPT`, `Report`, and `Base Paper`: For supporting materials.
