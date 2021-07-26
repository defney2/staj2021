# staj2021

The following files are the datasets that were acquired from the website Bot Repository in order to use as data sets to feed into the model:

"celebrity-2019.tsv",
"botwiki-2019.tsv",
"cresci-stock-2018.tsv",
"verified-2019.tsv",
"political-bots-2019.tsv",
"midterm-2018.tsv",
"pronbots-2019.tsv",
"crowdflower_results_detailed.tsv",
"gilani-2017.tsv",
"botometer-feedback-2019.tsv",
"cresci-rtbust-2019.tsv",
"vendor-purchased-2019.tsv"

Resource:https://botometer.osome.iu.edu/bot-repository/datasets.html

The file "twitter.py" is the Python script that was used in order to get the information of the accounts provided in the datasets and then gather them in a csv file. 

The file "dataframee2.csv" is the final database that was the product of the "twitter.py" script which served as the database for the project. 

The file "botpredictor.ipynb" is the exported Jupyter Notebook for the bot predictor. It takes in a screen_name and outputs its likelihood of being a bot.
The final output value "rf_probs0" is the likelihood of the account being a human. 
The final output value "rf_probs1" is the likelihood of the account being a bot. 

The "train2.py" file is the Python script of the contents of the Jupyter Notebook.
