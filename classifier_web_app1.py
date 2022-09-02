import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

###############################################################

from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler

###############################################################
st.title('Web App for Machine Learning Classification Problems')
st.write('Please visit the About page for more information by accessing the drop-down menu on the left sidebar.')

def main():
    activities=['EDA','Visualization','Model','About']
    options = st.sidebar.selectbox('Select Page Here', activities)
    







 ######################################################	EDA SELECT OPTION PAGE  #####################################################################

    if options =='EDA':
        st.subheader('Exploratory Data Analysis')
        
        data = st.file_uploader('Please upload dataset in CSV format', type=['csv']) # ALLOWS USER TO UPLOAD DATA
        if data is not None:
            st.success('Data Successfully Loaded')
        
        if data is not None: # IF DATA IS UPLOADED THESE OPTIONS WILL BE AVAILABLE
            df1 = pd.read_csv(data)
            st.subheader('Uploaded Dataset')
            st.dataframe(df1)

            if st.checkbox('Display Shape'): # DISPLAYS THE SHAPE OF DF1
                st.write('(rows, columns)',df1.shape)

            if st.checkbox('Display summary for complete dataset'): # DISPLAYS SUMMARY STATISTICS FOR DF1
                    st.write(df1.describe().T)

            st.title('Data Selection for EDA')
            st.info('Please select features/columns to explore') # INFO FOR SELECTING FEATURES

            if st.checkbox('Please select this box first and then select the columns you want to analyze from the the loaded dataset'): # ALLOWS USERS TO SELECT SPECIFIC COLUMNS AND CREATES DF2
                selected_columns = st.multiselect('Select Preferred Columns', df1.columns)
                df2 = df1[selected_columns]
                st.dataframe(df2)
                
                if st.checkbox('Display summary statistics for selected columns'): # DISPLAYS SUMMARY STATISTICS FOR THE SELECTED DATA
                    st.write(df2.describe().T)

                if st.checkbox('Display sum of null values per column'): # DISPLAY NULL VALUES PER COLUMNS AND ROWS
                    st.write(df2.isnull().sum())
                if st.checkbox('Display sum of null values per Row'):
                    st.write(df2.isnull().sum(axis=1))

                if st.checkbox('Display column data types'):
                    st.dataframe(df2.dtypes.astype(str))

                if st.checkbox('Display correlation matrix'):
                    st.write(df2.corr())








######################################################################## VISUALIZATION #################################################################


    elif options =='Visualization':
        st.subheader('Data Visualization')

        data = st.file_uploader('Please upload dataset in CSV format', type=['csv'])
        if data is not None:
        	st.success('Data Successfuly Loaded')
        	df1 = pd.read_csv(data)
        	st.subheader('Uploaded Dataset')
        	st.dataframe(df1)

        st.subheader('Select data columns to visualize')

        if st.checkbox('Please select this box first and then select the columns you want to analyze from the the loaded dataset'): # ALLOWS USERS TO SELECT SPECIFIC COLUMNS AND CREATES DF2
                selected_columns = st.multiselect('Select Preferred Columns for Visualization', df1.columns)
                df2 = df1[selected_columns]
                st.dataframe(df2)

        if st.checkbox('Display Correlation Heatmap'):
        	fig1 = plt.figure(figsize=(8,6))
        	sns.heatmap(df2.corr(), annot=True, cmap='GnBu')
        	st.pyplot(fig1)

        if st.checkbox('Display Pair Plot'):
        	if st.checkbox('Include a Hue in your Piar Plot?  Not required.'):
        		st.warning('Note: Certain columns will not work as a hue.  It is best to use hue for categorical columns.  For instance, use a categorical target column.')
        		selected_columns1 = st.selectbox('Select column to use as hue',df2.columns)
        		fig2 = sns.pairplot(df2, hue=selected_columns1)
        		st.pyplot(fig2)
        	else:
        		fig2 = sns.pairplot(df2)
        		st.pyplot(fig2)

        if st.checkbox('Display all selected columns on thier own figure. Boxplots'):
        	st.info('Note: This will display miltiple boxplots on their own visual figure.  This is because some columns can have different scales and that makes it difficult to visualize on one figure')
        	for i in df2.columns:
        		fig3 = plt.figure()
        		sns.boxplot(df2[i])
        		st.pyplot(fig3)

        if st.checkbox('Display all selected columns on one Boxplot figure'):
        	st.info('Note: Select or deselect columns from above to choose which columns to include in the visual')
        	fig4 = plt.figure()
        	sns.boxplot(data=df2, orient='h')
        	plt.tight_layout()
        	st.pyplot(fig4)

        if st.checkbox('Display Histogram'):
        	st.info('Note: Select or deselect columns from above to choose which columns to include in the visual')
        	fig5 = plt.figure()
        	sns.histplot(df2)
        	st.pyplot(fig5)








####################################################################### MODEL ########################################################################

    elif options == 'Model':
    	st.subheader('Model Building')
    	st.info('Please follow the bolded titles in the order as they appear. 1) Upload Data, 2) Select Dependent/Target Variable, 3) Select Independent/Feature Variables, 4) Train_Test_Split Options, 5) Scale Data (if needed), 6) Select Machine Learning Algorithm')
    	st.subheader('Upload Your Data')
		    	

    	data = st.file_uploader('Please upload dataset in CSV format', type=['csv'])
    	if data is not None:
    		st.success('Data Successfully Loaded.  Dataset below:')
    		df1 = pd.read_csv(data)
    		st.dataframe(df1)

    			############################# DEPENDENT VARIABLE SELECTION #######################

    	st.subheader('Dependent/Target Variable Selection')
    	st.warning('Note: Selecting the Dependent/Target variable is required.')
    	if st.checkbox('Please click to select dependent/target variable'):
    		selected_columns3 = st.selectbox('Select Preferred Dependent Variable', df1.columns)
    		y = df1[[selected_columns3]]


    			############################# INDEPENDENT VARIABLE SELECTION #######################

    	st.subheader('Independent/Feature Variables Selection')
    	st.warning('Note: Selecting the Independent/Feature variables are required.')
    	if st.checkbox('Please click to select independent/feature variables. Multiple selections allowed.'):
    		selected_columns2 = st.multiselect('Select Preferred Independent Variables. Re-click gray drop-down bar to select another variable.', df1.drop(y.columns, axis=1).columns)
    		# st.warning('Note: Do not select the dependent/target variable here. Doing so will cause errors')
    		X = df1[selected_columns2]



    			############################# TRIAN TEST SPLIT ####################
    	
    	st.subheader('Train Test Split')
    	st.write('Data will be split into training and testing sets')
    	testing_size = st.slider('Pick the proportion of data dedicated to testing. Default is 0.20',min_value=0.1, max_value=0.5, value=0.2)
    	random_state1 = st.slider('Pick the Random State.',min_value=0, max_value=5000, value=1000)
    	st.info('Note: Random State is used so the data is randomized but the state of randomization is known.  This is useful when testing various parameters or comparing different models.  They can be tested under the same random condition.')
    	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testing_size, random_state=random_state1)

    	if st.checkbox('Show Training and Testing Splits? (optional)'):
    		st.write('X_train set')
    		st.dataframe(X_train)
    		st.write('Dataset shape: ', X_train.shape)

    		st.write('X_test set')
    		st.dataframe(X_test)
    		st.write('Dataset shape: ',X_test.shape)

    		st.write('y_train')
    		st.dataframe(y_train)
    		st.write('Dataset shape: ',y_train.shape)

    		st.write('y_test')
    		st.dataframe(y_test)
    		st.write('Dataset shape: ',y_test.shape)





    			############################# SCALING INDEPENDENT VARIABLES #######################

    	st.subheader('Scaling Data')
    	st.info("Note: Scaling data might be needed when the scales of various independent/feature variables are largely different.  For example, income verse hours worked.  Income can be vary large compared to hours.  These large distances in values cause algorithms to place higher importance on larger values when in reality we want them to be equally important.  Also, scaling data can help meet some model's assumptions by normalizing the data.")
    	selected_scaler = None
    	if st.checkbox('Scale data? (optional)'):
    		scale_list = ['Pick a Scaler','Standardorize', 'MinMax']
    		selected_features_to_scale = st.multiselect('1) Select which independent/feature variables to scale and transform', X.columns)
    		selected_scaler = st.selectbox('2) Select a scaler',scale_list)

    			############################## Standard Scaler ####################################

    		if selected_scaler == 'Standardorize':
    			if selected_features_to_scale is not None:
    				data_to_scale = X_train[selected_features_to_scale]
    				scaler1 = StandardScaler()
    				scaled_X_train = scaler1.fit_transform(data_to_scale)
    				scaled_X_train = pd.DataFrame(scaled_X_train, columns=data_to_scale.columns, index=data_to_scale.index)
    				joined_scaled_with_non_scaled = pd.concat([scaled_X_train, X_train.drop(data_to_scale.columns, axis=1)], axis=1)

    								##### SCALED X_TRAIN ####
    				if st.checkbox('Show the data that was scaled and transformed?'):
    					st.dataframe(scaled_X_train)
    					st.write('Shape', scaled_X_train.shape)

    									##### JOINED SCALED WITH UNSCALED #####
    				if st.checkbox('Show entire X_train set with the scaled and transformed data?  Note: only useful if some of the data was not scaled.'):
    					st.write('Combined X_train set with the fit and transformed data')
    					st.dataframe(joined_scaled_with_non_scaled)
    					st.write('Shape: ',joined_scaled_with_non_scaled.shape)

    				scaled_X_test = scaler1.transform(X_test[selected_features_to_scale])
    				scaled_X_test = pd.DataFrame(scaled_X_test, columns=selected_features_to_scale, index=X_test.index)
    				joined_scaled_with_non_scaled_test = pd.concat([scaled_X_test, X_test.drop(scaled_X_test.columns, axis=1)], axis=1)
    				
    				if st.checkbox('Show entire X_test set with the transformed data?  Note: only useful if some of the data was not scaled.'):
    					st.write('Combined X_test set with transformed data')
    					st.dataframe(joined_scaled_with_non_scaled_test)
    					st.write('Shape: ',joined_scaled_with_non_scaled_test.shape)

    			############################## Min Max Scaler ####################################


    		if selected_scaler == 'MinMax':
    			if selected_features_to_scale is not None:
    				data_to_scale = X_train[selected_features_to_scale]
    				scaler2 = MinMaxScaler()
    				scaled_X_train = scaler2.fit_transform(data_to_scale)
    				scaled_X_train = pd.DataFrame(scaled_X_train, columns=data_to_scale.columns, index=data_to_scale.index)
    				joined_scaled_with_non_scaled = pd.concat([scaled_X_train, X_train.drop(data_to_scale.columns, axis=1)], axis=1)
    				
    				if st.checkbox('Show the data that was scaled and transformed?'):
    					st.dataframe(scaled_X_train)


    				if st.checkbox('Show entire X_train set with the scaled and transformed data?  Note: only useful if some of the data was not scaled.'):
    					st.write('Combined X_train set with the fit and transformed data')
    					st.dataframe(joined_scaled_with_non_scaled)

    				scaled_X_test = scaler2.transform(X_test[selected_features_to_scale])
    				scaled_X_test = pd.DataFrame(scaled_X_test, columns=selected_features_to_scale, index=X_test.index)
    				joined_scaled_with_non_scaled_test = pd.concat([scaled_X_test, X_test.drop(scaled_X_test.columns, axis=1)], axis=1)
    				
    				if st.checkbox('Show entire X_test set with the transformed data?  Note: only useful if some of the data was not scaled.'):
    					st.write('Combined X_test set with transformed data')
    					st.dataframe(joined_scaled_with_non_scaled_test)


    					####### joined_scaled_with_non_scaled_test IS THE COMBINED DATASET FOR TESTING THE MODEL #######
    					####### joined_scaled_with_non_scaled IS THE COMBINED DATASET FOR TRAINING THE MODEL ###########

    	st.subheader('Select the Machine Learning Algorithm')
    	algor_list = ['SVC', 'Random Forest Classificer', 'KNN', 'LogisticRegression']
    	st.info('Note: There are many different hyperparameters that could have been included but this project is a simple test to show what is possible and not all inclusive.')
    	algorithm = st.selectbox('Select from the list of machine learning algorithms', algor_list)



    		############################## GET PARAM FUNCTION #########################################


    	def add_param(name_clf):
    		params=dict()

    		if name_clf=='SVC':
    			st.info('Below are the hyperparameters that can be adjusted')
    			C = st.number_input('Input the C parameter', min_value=0.0001, max_value=100.0, value=0.001, step=0.0001)
    			st.write('C value selected:', C)
    			kernel_list = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
    			kernel = st.selectbox('Select the kernel',kernel_list)
    			gamma = st.number_input('Input the gamma parameter', min_value=0.0001, max_value=100.0, value=0.001, step=0.0001)
    			st.write('Gamma value selected: ', gamma)
    			params['C'] = C
    			params['kernel'] = kernel
    			params['gamma'] = gamma

    		if name_clf=='Random Forest Classificer':
    			st.info('Below are the hyperparameters that can be adjusted')
    			n_estimators = st.slider('Select the n_estimators parameter', min_value=25, max_value=500, value=100, step=25)
    			params['n_estimators'] = n_estimators

    		if name_clf=='KNN':
    			st.info('Below are the hyperparameters that can be adjusted')
    			n_neighbors = st.slider('Select the n_neighbors parameter', min_value=1, max_value=50, value=9, step=1)
    			params['n_neighbors'] = n_neighbors

    		if name_clf=='LogisticRegression':
    			st.info('Below are the hyperparameters that can be adjusted')
    			max_iter = st.slider('Select the max_iter parameter', min_value=100, max_value=10000, step=100, value=10000)
    			C = st.number_input('Select the C parameter', min_value=0.001, max_value=10.0, step=0.001, value=1.0)
    			st.write('C value selected:', C)
    			params['max_iter'] = max_iter
    			params['C'] = C

    		return params

    	params = add_param(algorithm)

    		######################### PASSING ALGORITHM AND PARAMS ##################################

    	def get_classififier(name_clf,params):
    		clf = None
    		if name_clf == 'SVC':
    			clf = SVC(C = params['C'], kernel = params['kernel'], gamma = params['gamma'])

    		elif name_clf == 'Random Forest Classificer':
    			clf = RandomForestClassifier(n_estimators = params['n_estimators'], n_jobs = -1)

    		elif name_clf == 'KNN':
    			clf = KNeighborsClassifier(n_neighbors = params['n_neighbors'], n_jobs = -1)

    		elif name_clf == 'LogisticRegression':
    			clf = LogisticRegression(max_iter = params['max_iter'], C = params['C'], n_jobs = -1)

    		else:
    			st.warning('Please select a machine learning algorithm')

    		return clf



    	clf = get_classififier(algorithm, params)

 			######################################### MODEL VAL EVAL ####################################
    	st.subheader('Model Validation & Evaluation')
    	st.info('Note: Repeated Stratified Cross Validation is being used.')
    	
    	st.write('Cross Validation Options.  Leave at default values if unsure.')
    	splits = st.slider('Select n_splits to divide test data for evaluation', min_value=2, max_value=20, value=5)
    	repeats = st.slider('Select n_repeats to repeat the cross validation', min_value=1,max_value=20, value=10)
    	random_state2 = st.slider('Select the random state', min_value=0, max_value=5000, value=1000 )    	
    	cv = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeats, random_state=random_state2)


    	if selected_scaler == None:
    		st.write(clf)
    		score = cross_val_score(estimator=clf, X = X_train, y = y_train, cv = cv, scoring = 'accuracy', n_jobs=-1)
    		st.write('Cross Validation Mean Score: ', round(score.mean()*100,3),'%')
    		st.write('Cross Validation Standard Deviation +- Score:', round(score.std()*100,3),'%')

    		st.write('Validation Scores per Iteration of Cross Validation:')
    		st.dataframe(score)
    		fig = plt.figure()
    		plt.title('Distribution of Validation Scores')
    		sns.histplot(score, bins=6, kde=True)
    		st.pyplot(fig)
    		
    		st.warning('Before moving on to Model Testing, consider the scaler used (if used) and the hyperparameters selected.  Make all FINAL changes before moving on to the model testing phase.')
    		
    	else:
    		st.write(clf)
    		score = cross_val_score(estimator=clf, X = joined_scaled_with_non_scaled, y = y_train, cv = cv, scoring = 'accuracy', n_jobs=-1)
    		st.write('Cross Validation Mean Score: ', round(score.mean()*100,3),'%')
    		st.write('Cross Validation Standard Deviation +- Score:', round(score.std()*100,3),'%')

    		st.write('Validation Scores per Iteration of Cross Validation:')
    		st.dataframe(score)
    		fig = plt.figure()
    		plt.title('Distribution of Validation Scores')
    		sns.histplot(score, bins=6, kde=True)
    		st.pyplot(fig)

    		st.warning('Before moving on to Model Testing, consider the scaler used (if used) and the hyperparameters selected.  Make all FINAL changes before moving on to the model testing phase.')


           ############################### MODEL TESTING #####################################
 
    	st.subheader('Model Testing and Scoring')
    	st.warning('DO NOT TUNE PARAMETERS BASED ON THESE RESULTS.  TUNING OCCURS BEFORE HAVING ACCESS TO TESTING DATA.  TESTING DATA IS MEANT TO BE UNSEEN.  ANY MODEL PARAMETERS CHANGED BASED ON THE MODEL TESTING SCORES INVALIDATES THE ENTIRE MODEL DUE TO THE BIAS INTRODUCED.')
    	if st.checkbox('Test Model?'):

    		if selected_scaler == None:
    			clf.fit(X_train,y_train)
    			y_pred = clf.predict(X_test)
    			st.write('Model Accuracy Score: ', round(accuracy_score(y_test,y_pred)*100,3),'%')
    			st.write('Confusion Matrix:')
    			st.dataframe(confusion_matrix(y_test,y_pred))
    			st.write('Test Values verse Predicted Values: ')
    			y_pred = pd.DataFrame(y_pred, columns=['predicted values'], index=y_test.index)
    			test_pred = pd.concat([y_test, y_pred], axis=1)
    			st.dataframe(test_pred)

    		else:
    			clf.fit(joined_scaled_with_non_scaled, y_train)
    			y_pred = clf.predict(joined_scaled_with_non_scaled_test)
    			st.write('Model Accuracy Score: ', round(accuracy_score(y_test,y_pred)*100,3),'%')
    			st.write('Confusion Matrix:')
    			st.dataframe(confusion_matrix(y_test,y_pred))
    			st.write('Test Values verse Predicted Values: ')
    			y_pred = pd.DataFrame(y_pred, columns=['predicted values'], index=y_test.index)
    			test_pred = pd.concat([y_test, y_pred], axis=1)
    			st.dataframe(test_pred)




####################### ABOUT ###################################

    else:
    	# options == 'About'
    	st.subheader('Purpose of this Web Application')
    	st.write('Thank you for stopping by and viewing my web application.  This web application is an example of exploring data, visualizing data, and modeling data.  This project is meant to show recruiters and hiring managers my ability to create and deploy machine learning models for stakeholders.  This example is a general example, but it could be tailored to the needs of the stakeholders.')
    	st.subheader('About the Creator')
    	st.write('My name is Brandon Johnson.  I have 10 years of experience as an observational researcher and analyst for the federal government.  I am finishing a bachelor’s degree in Business Analytics within the next year.  I am also concurrently training to be a Data Scientist by taking courses tailored for that role.  This web application is an example of one of the many hands-on projects I have completed as a Data Scientist student.  The skills needed to produce this project were: Python coding, data analysis, machine learning modeling, model deployment, web application design, and so on. Check out my LinkedIn for more information about the creator: www.linkedin.com/in/brandon-johnson-09645ba9')
    	st.subheader('Importing Data')
    	st.write('The dataset used to build this program is called the Iris dataset.  This dataset can be accessed and downloaded here: https://github.com/SPDA36/WebApp1/blob/main/iris.csv')
    	st.subheader('Web Application Usage')
    	st.write('This web application can be used for general machine learning classification.  You are welcome to use it and share it with others.  Keep in mind that in a real business environment, this would be tailored for the end user.')




if __name__ == '__main__':
    main()
