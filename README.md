# Dat4-BI-Exam

## Data sources:

https://www.kaggle.com/datasets/sazidthe1/data-science-salaries

https://www.kaggle.com/datasets/yusufglcan/country-data

Business Model: Global Career Advisors (måske skift senere)

Mission:
To empower students and professionals in making informed decisions about their career paths by providing personalized advice on optimal workplaces around the world.

Marketing Stragety:
We provide a global workspace intelligence, finding the most appropriate build for an effective job with an effective income.
We will in this data figure out what type of workmode is better (eg. remote, on-site og hybrid) as well as what sort of jobs have done best over the years.

Hypothesis:
We believe that the salary is affected by the job type and the experience level and the location of the job because of GDP per capita.

Tasks:

[] Create different questions for what to research: Ex. as education institute I want to know what jobs are available in my country. Ex. as employee I want to know how much I should earn
[] Make clustering based on questions (No PCA)
[] Use cluster results to create new column based on cluster results
[] Use new columns to classify and predict (cluster number can be used as prediction point)
[] Fetch country data (population statistics, etc...) and formulate questions which both data can be used to answer: Ex. Does BNP have an effect on salary?
[] Use new data to regress

## Questions

1. As an Employee, looking at the data, which factors factors affect my salary?

// how much more can I expect to earn based on experience level, work model, job title, country?

DataProcessing

2. Does GDP per capita of the Country have an effect on salary?

DataProcessing

Lav GDP = Større chance for du får en lav løn
Højere GDP = Større chance for du får en højere løn, men stadig mange som får en mindre løn, mange faktorer spiller ind pga. forskelllige jobs, erfaring etc.
Højest GDP = Vi har ikke nok data punkter til at kunne sige meget om det.

3. As an Employee, I want to know how much I should earn.

Regression

4. What parameters affect the salary the most?

Clustering
Classification - Decision Tree
