def dropEmptyCols(df,emptinessThreshold=1) :
    
    """return an input pandas.DataFrame without its empty columns
    
    inputs
    ------
    df : pandas.DataFrame object
    
    optional inputs 
    ---------------
    emptinessThreshold : float, default : = 1. Gives the limit threshold of emptiness. 
    if >=threshold, column is dropped. 
    
    returns
    -------
    dfWithoutEmptyCols : pandas.DataFrame object, the same one without its empty columns
    
    """
    #select columns with percentage of null values > threshold
    emptyColsNames=df.isna().mean().loc[df.isna().mean()>=emptinessThreshold].index 
    
    #drop selected columns
    dfWithoutEmptyCols=df.copy()
    dfWithoutEmptyCols = dfWithoutEmptyCols.drop(columns=emptyColsNames)
    
    return dfWithoutEmptyCols




def getCombiOfCol(listOfCols) :
    from itertools import combinations
    dictOfCombis = {}
    for n in range(len(listOfCols)) :
        listOfCombis=[]
        for combi in list(combinations(listOfCols,len(listOfCols)-n)) :
            listOfCombis.append(list(combi))
        dictOfCombis[len(listOfCols)-n]=listOfCombis
    return dictOfCombis


def myBoxPlot(df,numColName,direction="h",ax=None) :
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.boxplot(data=df,
                x=numColName if direction=="h" else None,
                y=numColName if direction=="v" else None
                ,ax=ax)


def myDescribe(dataframe) :
    
    '''displays a Pandas .describe with options : quantitaves columns, qualitatives columns, all columns.
    If a dict is given as an input : {"df1Name" : df1, "df2Name" : df2, etc.}, the one can choose the dataframe
    
    parameters :
    ------------
    dataframe : Pandas dataframe or a Dict
    
    '''
    
    
    import ipywidgets as widgets # import library
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # def main function
    def myDescribeForOnlyOneDf(df) :
        
        # if df is a dictionnary key, we get its value
        if type(df)==str :
            df=dataframe[df]
        
        
        # widget with .describe() display options
        widDescribe=widgets.RadioButtons(options=["quantitative","qualitative","all"], 
                                         value="all",
                                         description="Which features :",
                                         disabled=False,
                                        style={'description_width': 'initial'}
                                        )
        
        # widget to select column 
        widColList=widgets.Dropdown(options={"all" : None}|{col : col for col in list(df.columns)},
                                  value=None,
                                  description="Which column :",
                                  disabled=False,
                                  style={'description_width': 'initial'}
                                  )
        
        def handle_widDescribe_change(change): # linking col dropdwn with the type of describe display option
            if change.new == "qualitative" :
                widColList.options={"all" : None}|{col : col for col in list(df.select_dtypes(["O","category"]).columns)}
            if change.new == "quantitative" :
                widColList.options={"all" : None}|{col : col for col in list(df.select_dtypes(["float64","float32","float16","int64","int32","int16","int8","uint8"]).columns)}
            if change.new == "all" :
                widColList.options={"all" : None}|{col : col for col in list(df.columns)}
        
        widDescribe.observe(handle_widDescribe_change,'value')
        
        # sub function used in final output
        def describeFunct(df, whichTypes,columnName=None) :
            if whichTypes=="qualitative" :
                include=["O","category"]
                exclude=["float64","float32","float16","int64","int32","int16","int8","uint8"]
            elif whichTypes=="quantitative" :
                include=None
                exclude=None
            elif whichTypes=="all" :
                include="all"
                exclude=None
            if columnName :
                df=df[[columnName]]
            describeTable=df.describe(include=include, exclude=exclude)
            # add dtypes
            describeTable.loc["dtype"]=describeTable.apply(lambda s : df[s.name].dtype).values.tolist()
            describeTable.loc["%NaN"]=describeTable.apply(lambda s : (round(df[s.name].isna().mean()*100,1)).astype(str)+"%").values.tolist()
            describeTable=pd.concat([describeTable.iloc[-1:],describeTable.iloc[:-1]])
            
            # descide which kind of display
            if columnName and df[columnName].dtype.kind not in 'O':
                fig, (ax1,ax2) = plt.subplots(1,2,width_ratios=[1,4],figsize=(12,4))
                ax1.table(cellText=describeTable.values,
                         rowLabels=describeTable.index,
                         bbox=[0,0,1,1],
                         colLabels=describeTable.columns)
                ax1.axis(False)

                myBoxPlot(df=df,numColName=columnName,direction="h",ax=ax2)
                plt.show()
                
            else :
                display(describeTable)
        
        # output
        out = widgets.interactive_output(describeFunct, {"df" : widgets.fixed(df),
                                                    "whichTypes" : widDescribe ,
                                                         "columnName" : widColList
                                                                   }
                                        )
        display(widgets.HBox( [widDescribe,widColList]),out)
    
    # if input is a dataframe, use above function
    if type(dataframe)!=dict :
        myDescribeForOnlyOneDf(dataframe)
    
    # if input is a dict, add a widget to select a dataframe
    else :
        
        widDfList=widgets.Dropdown(options=list(dataframe.keys()),
                                  value=list(dataframe.keys())[0],
                                  description="Which dataframe :",
                                  disabled=False,
                                  style={'description_width': 'initial'}
                                  )
        
        out = widgets.interactive_output(myDescribeForOnlyOneDf, {"df" : widDfList
                                                                   }
                                        )
        display(widDfList,out)

def generateDfsToTestImput (df) :
    '''       
    Generate two dataframes to test missing values imputation techniques : 
    - one dataframe with all complete rows of the input dataframe
    - the same dataframe, but with artificial missing values    
    
    if imputaters are distance based, better to use an already scaled dataframe
    
    parameters
    ----------
    df : dataframe
        
    '''
    
    # create a dataframe with the non null rows of df

    dfComplete = df.copy() # create a copy
    dfComplete = dfComplete.loc[dfComplete.notna().all(axis=1)] # keep non null rows only 

    # create a second dataframe with null values
    
    # use a function to compute the combinations of the df's columns 
    combis=getCombiOfCol(df.columns.tolist())
    
    # create copys
    dfCompTransit = dfComplete.copy()
    dfTransit = df.copy()
    
    # list to store parts of futur incomplet dataframe
    dfIncompleteList =[]
    
    # first step - create a part with complet rows
    maskFull = df.notna().all(axis=1) # mask for complete rows of df
    fracFull = maskFull.mean() # % of complete rows in df
    
    dfTransit = dfTransit.loc[~maskFull] # exculde these complete rows from dfTransit
    
    filterFull = dfCompTransit.sample(frac=fracFull,random_state=0) # create a sample of dfCompTransit with the same proportion
    dfIncompleteList.append(filterFull) # add this complete part to the list
    
    idxFull = filterFull.index # store index
    dfCompTransit = dfCompTransit.loc[~dfCompTransit.index.isin(idxFull)] # exclude these rows of dfCompTransit
    
    # then - iterate on combinations of columns to create missing values
    for i,listOfcombis in enumerate(combis.values()) :
        for j,combi in enumerate(listOfcombis) :

            maskDf=dfTransit[combi].isna().all(axis=1) # mask of empty rows for this combination in dfTransit
            fracCombiDf=maskDf.mean() # proportion

            dfTransit=dfTransit.loc[~maskDf] # exclude these empty rows from dfTransit
            
            combiFilter=dfCompTransit.sample(frac=fracCombiDf,random_state=100*(i+1)+10*(j+1)) # create a sample with the same proportion
            combiFilter[combi]=np.nan # fill this sample with NaNs for columns of combi
            dfIncompleteList.append(combiFilter) # add this part to the list
            indCombi=combiFilter.index # store the index
            dfCompTransit=dfCompTransit.loc[~dfCompTransit.index.isin(indCombi)] # exclude these rows from dfCompTransit
            if len(dfCompTransit) == 0 : # stop if dfCompTransit is empty
                break
    
    # create dfIncomplete from the list
    dfIncomplete = pd.concat(dfIncompleteList).sort_index()
    
    return dfComplete , dfIncomplete


def myKNNCategoricalImputer (df,catCol,k) :
    
    '''       
    from a given dataframe and a given categorical feature name, 
    returns the same dataframe with the categorical feature missing values imputed.
    
    Imputation porcess uses the KNN classifier from scikit learn.
    
    
    
    parameters
    ----------
    df : dataframe
    catCol : string, name of the categorcial feature
    k : int, hyperparameter for knn classifier
    
    return
    ------
    resultDf : same dataframe, with catCol missing values imputed
    
    '''
    
    # import library
    from sklearn import neighbors
    
    # create a copy
    workDf=df.copy()
    
    # if not, set catCol feature dtype to "category"
    workDf[catCol]=workDf[catCol].astype("category")
    
    # store catCol mode
    modeCatCol = workDf[catCol].mode()[0]
    
    # split workDf into "train" and "to imputed" dataframes
    workDfTrain = workDf.loc[workDf[catCol].notna()]
    workDfToImp = workDf.loc[workDf[catCol].isna()]

    # store the numerical features names in a list, for knn algo
    knnCols = [col for col in workDfTrain.select_dtypes("float64").columns.tolist()]
    
    # use getCombiOfCol function to create a dictionnary of all numerical columns combinations
    combisKnnCols = getCombiOfCol(knnCols)
    
    # create the output dataframe
    resultDf=workDf.copy()
    
    # imputation 
    # itirate on lists of columns combinations :
    for n,listOfCombi in combisKnnCols.items() : # for each list of list of columns with a lenght of n elements

        for ColList in listOfCombi : # for each columns combination
            workDfToImpColList = workDfToImp.loc[workDfToImp[ColList].notna().all(axis=1)] # filter on rows with all features values completed

            if len(workDfToImpColList)==0 :
                continue # if no observations in this filtered dataframe, continue
                
            idxToImpColList = workDfToImpColList.index # store index

            workDfTrainColList = workDfTrain.loc[workDfTrain[ColList].notna().all(axis=1)] # filter the same way on the train df

            if len(workDfTrainColList)>=k : # if the number of neighbors in workDfTrainColList is equal or above to our given k
                knnColList = neighbors.KNeighborsClassifier(n_neighbors=k) # intanciate  knn classifier
                knnColList.fit(workDfTrainColList[ColList],workDfTrainColList[catCol]) # train it on our filtered train set

                imputationsColList=knnColList.predict(workDfToImpColList[ColList]) # predict catCol values on our "to imput" filtred df

            else : # if the number of neighbors is below k
                imputationsColList=workDfToImpColList[catCol].fillna(value=modeCatCol).values # we fill with the mode

            resultDf.loc[idxToImpColList,catCol]=imputationsColList # put results on resultDf

            workDfTrain = resultDf.loc[resultDf[catCol].notna()] # remove rows from workDfTrain
            workDfToImp = resultDf.loc[resultDf[catCol].isna()] # remove rows from workDfToImp

            if len(workDfToImp) == 0 :
                break # break if workDfToImp is empty
    return resultDf


def BestKforMyKNNCatImputer(dfComplete, dfIncomplete, catCol, rangeOfK=(2,10)) :
    
    ''' 
    generate a 2 columns dataframe with r2 and Root Mean Squared Error for each numerical feature
    
    
    parameters
    ----------
    dfComplete : dataframe only with no missing values
    dfIncomplete : same dataframe, with missing values
    catCol : name of the categorical columns to imput
    
    optional parameters
    -------------------
    rangeOfK : tuple, interval of Ks to test on
    
    returns
    -------
    resultTabPercentError : dataframe with
        Ks in index 
        a column with the percentage of errors
    '''
        
    
    # create dataframe to store the mesure for each k
    resultTabPercentError=pd.DataFrame(columns=["%Error_"+catCol])
       
    # for each K, run the myKNNCategoricalImputer function on catCol
    for K in range(rangeOfK[0],rangeOfK[1]+1) :
                      
        # generate a seriesImput for catCol
        dfImput=myKNNCategoricalImputer(dfIncomplete,catCol,k=K)
        seriesImput=dfImput.loc[dfIncomplete[catCol].isna(),catCol]

        seriesSoluce=dfComplete.loc[dfIncomplete[catCol].isna(),catCol] # the real values for catCol
       
        # put in the main tab
        resultTabPercentError.loc["k="+str(K)]=(seriesImput!=seriesSoluce).mean()

  
    return resultTabPercentError


# function to get in a dataframe the results of imputation for a specific feature

def getSoluceColAndGuessCol (col, dfComplete, dfIncomplete, dfImput) :
    
    ''' 
    generate a 2 columns dataframe with the results of imputation for a specific feature
    
    
    parameters
    ----------
    col : string, feature's name
    dfComplete : dataframe only with no missing values
    dfIncomplete : same dataframe, with missing values
    dfImput : same dataframe, with the missing values imputed
    
    returns
    -------
    SoluceAndGuessTab : dataframe with
        only original missing values indexes, 
        a column of expected values 
        a columns of imputed values
    '''
    
    # get missing values index
    colNanIndex = dfIncomplete.loc[dfIncomplete[col].isna()].index # get nan values index
    
    # create tab
    SoluceAndGuessTab = pd.DataFrame() # initiate dataframe
    SoluceAndGuessTab[col+"_SOLUCE"]=dfComplete.loc[colNanIndex,col] # create expected values column
    SoluceAndGuessTab[col+"_IMPUT"]=dfImput.loc[colNanIndex,col] # create imputed values column
    
    return SoluceAndGuessTab


def resultsNumImput(dfComplete, dfIncomplete, dfImput) :
    
    ''' 
    generate a 2 columns dataframe with r2 and Root Mean Squared Error for each numerical feature
    
    
    parameters
    ----------
    dfComplete : dataframe only with no missing values
    dfIncomplete : same dataframe, with missing values
    dfImput : same dataframe, with the missing values imputed
       
    returns
    -------
    resultTab : dataframe with
        numerical features in index 
        a column with RMSE 
        a column with R2
    '''
        
    # list of numerical features
    numFeaturesNames = dfComplete.select_dtypes("float64").columns.tolist()
      
    # initiate tab
    resultTab = pd.DataFrame(columns=['RMSE','R2'],index=numFeaturesNames,dtype="float64")
    
    # compute RMSE and R2 for each column
    for col in numFeaturesNames :
        
        colNanIndex = dfIncomplete.loc[dfIncomplete[col].isna()].index # get nan values index
        
        # get the columns
        soluceCol = dfComplete.loc[colNanIndex,col]
        imputCol  = dfImput.loc[colNanIndex,col]
        
        # compute RMSE and R2, and put them in tab
        RMSECol = np.sqrt(mean_squared_error(soluceCol,imputCol))
        resultTab['RMSE'].loc[col]=RMSECol
        
        R2Col = r2_score(soluceCol,imputCol)
        resultTab['R2'].loc[col]=R2Col
        
    return resultTab


def BestKNNImputer(dfComplete, dfIncomplete, rangeOfK=(2,10)) :
    
    ''' 
    generate a 2 columns dataframe with r2 and Root Mean Squared Error for each numerical feature
    
    
    parameters
    ----------
    dfComplete : dataframe only with no missing values
    dfIncomplete : same dataframe, with missing values
    dfGuess : same dataframe, with the missing values imputed
    
    optional parameters
    -------------------
    nameOfTest : string, name of the imputation test
    
    returns
    -------
    resultTab : dataframe with
        numerical features in index 
        a column with RMSE 
        a column with R2
    '''
    
    # list of numerical features
    numFeaturesNames = dfComplete.select_dtypes("float64").columns.tolist()
    
    # create dataframes to store the mesures for each k
    resultTabRMSE=pd.DataFrame(index=numFeaturesNames,dtype="float64")
    resultTabR2=pd.DataFrame(index=numFeaturesNames,dtype="float64")
    
    # create a dataframe to store the mean of RMSEs and the mean of R2s for each k
    resultTabGlobal=pd.DataFrame(columns=["k="+str(i) for i in range(rangeOfK[0],rangeOfK[1]+1)], 
                                 index=["Mean_of_RMSEs","Mean_of_R2s"],dtype="float64")
    
    for k in range(rangeOfK[0],rangeOfK[1]+1) :
        
        KNN_imputer=KNNImputer(n_neighbors=k,missing_values=np.nan) # initiate knn imputer for this k
        
        # generate a dfGuess
        dfImputK=pd.DataFrame(
            KNN_imputer.fit_transform(dfIncomplete[numFeaturesNames]), # fit transform
            columns=numFeaturesNames,
            index=dfIncomplete.index
        )
        
        # use previous function and generate a resultsNumImput(dfComplete, dfIncomplete, dfImput) :
        resultTabK=resultsNumImput(
            dfComplete=dfComplete, 
            dfIncomplete=dfIncomplete, 
            dfImput=dfImputK, 
        )
        
        # put in the main tabs
        resultTabRMSE["RMSE"+"_"+"k="+str(k)]=resultTabK["RMSE"]
        resultTabR2["R2"+"_"+"k="+str(k)]=resultTabK["R2"]
        
       
        resultTabGlobal["k="+str(k)].loc["Mean_of_RMSEs"]=resultTabK["RMSE"].mean()
        resultTabGlobal["k="+str(k)].loc["Mean_of_R2s"]=resultTabK["R2"].mean()
    
    # give for each numerical feature the k with best RMSE and R2
    resultTabRMSE["best_k_per_feature"]=resultTabRMSE.idxmin(axis=1).str.split("=").str[-1]
    resultTabR2["best_k_per_feature"]=resultTabR2.idxmax(axis=1).str.split("=").str[-1]
    
    # give, globally, for Mean_of_RMSEs and for Mean_of_R2s, the best k
    resultTabGlobal.loc["Mean_of_RMSEs","best_k_on_average"]=resultTabGlobal.loc["Mean_of_RMSEs"].idxmin(axis=0).split("=")[-1]
    resultTabGlobal.loc["Mean_of_R2s","best_k_on_average"]=resultTabGlobal.loc["Mean_of_R2s"].astype("float64").idxmax(axis=0).split("=")[-1]
  
    return resultTabRMSE,resultTabR2,resultTabGlobal


def ShowResultsNumImputsSolo (measuresTabs,
                              testName,
                              measureType,
                              testsNamesList=None,
                              dictPalette=None,
                              figsize=(8,6)
                             ) :

    '''draw 1 barplot  of numerical features imputation test results : RMSE or R2

    parameters
    ----------
    measuresTabs : dataframe or list of df, output of the resultsNumImput() function
    testName : string , name of the imputation test
    measureType : string, type of imputation perf measure - "RMSE" or "R2"

    optionnal
    ---------
    testsNamesList : list of tests names
    dictPalette : dictionnary, with features names for keys and colors for values
    figsize : tuple or list, dimensions of the figure
        
    returns
    -------
    a barplot : RMSE or R2, for each feature
    '''
        
    # visualisation of imputation result for a specific test and a specific measure
    sns.set_theme()
    # convert measuresTabs and testNames if not lists
    if type(measuresTabs)!=list :
        measuresTabs=[measuresTabs]
    if testsNamesList and type(testsNamesList)!=list :
        testsNamesList=[testsNamesList]
    # select tab from measuresTabs
    if testsNamesList :
        tab=measuresTabs[testsNamesList.index(testName)]
    else :
        tab=measuresTabs[0]

    # set main parameters for the visualisation
    sns.set_theme()
    fig,axs=plt.subplots(1, 
                         figsize=figsize,
                         ) 
    fig.suptitle("Résultats des tests d'imputation avec "+testName+" - valeurs de "+measureType,fontsize=12) 
    xlabels=[col for col in measuresTabs[0].index] # set labels : features names

    
    # create barplot
        
    axs.bar(x=[1+k for k in range(len(xlabels))], # one bar for each feature
            height=tab[measureType], # the measure value
            width=0.7,
            color=[dictPalette[col] 
                   if dictPalette 
                   else "r" 
                   for col in xlabels], # if each feature has a specific color, we use it
            ec="k", # black bar edges 
            alpha=1
           )

    axs.set_xlim(0,len(xlabels)+1) # set x axis min and max
    axs.set_xticks([k+1 for k in range(len(xlabels))]) # set positions of ticks
    axs.set_xticklabels(xlabels,rotation=45,ha="right") # features names for ticklabels
    axs.set_ylabel(measureType)
    if measureType=="RMSE" :
        axs.set_ylim(bottom=0,top = max([tab["RMSE"].max() for tab in measuresTabs])*1.1) # set y axis min max for RMSE barplot
    if measureType=="R2" :
        axs.set_ylim(bottom=0,top=1) # set y axis min for R2 bar plot
    
    plt.show()



def ShowResultsNumImputsWID (measuresTabs,testsNamesList,dictPalette=None,figsize=(6,5)) :

    '''draw 1 barplot  of numerical features imputation test results : RMSE or R2

    parameters
    ----------
    measuresTabs : dataframe or list of df, output of the resultsNumImput() function
    testsNamesList : list of tests names (can be a string if only one test)

    optionnal
    ---------
    dictPalette : dictionnary, with features names for keys and colors for values
    figsize : tuple or list, dimensions of the figure
        
    returns
    -------
    a barplot : with RadioButtons to select :
        - a test
        - a measure type (RMSE or R2)
    '''
    
    # convert measuresTabs and testNames if not lists
    if type(measuresTabs)!=list :
        measuresTabs=[measuresTabs]
    if type(testsNamesList)!=list :
        testsNamesList=[testsNamesList]

    
    # create ipywidgets RadioButtons
    import ipywidgets as widgets # import library
    
    
    widMeasure=widgets.RadioButtons(options=measuresTabs[0].columns, 
                                     value=measuresTabs[0].columns[0],
                                     description="Performance measure :",
                                     disabled=False,
                                    style={'description_width': 'initial'}
                                    )
    
    widTest=widgets.RadioButtons(options=testsNamesList,
                                     value=testsNamesList[-1],
                                     description="Imputation test :",
                                     disabled=False,
                                 style={'description_width': 'initial'}
                                    )

    ui=widgets.HBox([widTest,widMeasure]) # horizontal user interface 
    
    # use defined widgets on ShowResultsNumImputsSolo()
    out = widgets.interactive_output(ShowResultsNumImputsSolo, {"measuresTabs" : widgets.fixed(measuresTabs),
                                                                "testName" : widTest ,
                                                                "measureType" : widMeasure,
                                                                "testsNamesList" : widgets.fixed(testsNamesList),
                                                                "dictPalette" : widgets.fixed(dictPalette),
                                                                "figsize":widgets.fixed(figsize)
                                                               }
                                    )
    
    display(ui,out)



def ShowResultsNumImputsMulti (measuresTabs,testsNamesList,dictPalette=None) :

    # visualisation of imputation tests measures

    '''draw barplots  of numerical features imputation tests results (one for RMSE, one for R2)

    parameters
    ----------
    measuresTabs : dataframe or list of dataframes, output(s) of the resultsNumImput function
    testsNamesList : string or list of strings, name(s) of the imputation test(s)
    
    optionnal
    ---------
    dictPalette : dictionnary, with features names for keys and colors for values
        
    returns
    -------
    a figure with 2 or 3 axes
        - 2 barplots : one for RMSE and one for R2
        - if several tests results, draw a legend on the 3rd axe
    '''
    
    # convert measuresTabs and testsNamesList if not lists
    if type(measuresTabs)!=list :
        measuresTabs=[measuresTabs]
    if len(measuresTabs)==1 :
        several=0 # store the information, we have only one test
    else :
        several=1 # store the information, we have several tests, we will need a legend
    if type(testsNamesList)!=list :
        testsNamesList=[testsNamesList]

    
    # store  the number of different tests
    m=len(measuresTabs[0].columns)

    # set main parameters for the visualisation
    sns.set_theme()
    fig,axs=plt.subplots(1,m+several, # an additionnal axe if we need to create a legend
                         figsize=(16,8),
                         gridspec_kw={'width_ratios': [3 if k!=m else 1 for k in range(m+several)]} # if a legend, smaller
                        ) 
    if several==0 : # if we have only one test, we can put its name in the main title
        fig.suptitle("Résultats des tests d'imputation avec "+testsNamesList[0],fontsize=20) 
    else : # if we have several tests, the different names will be in the legend
        fig.suptitle("Résultats des tests d'imputation avec chaque méthode",fontsize=20)
    xlabels=[col for col in measuresTabs[0].index] # set labels : features names

    # set width of the bars
    l = len(measuresTabs)
    myWidth = 0.7/l
    # create an offset for each bar
    xOffset = 1-(l-1)*myWidth/2

    # create barplots for RMSE and R2
    for i,tab in enumerate(measuresTabs) : # we iterate on tests
        for j,measure in enumerate(tab.columns) : # we iterate on measures

            axs[j].bar(x=[k+xOffset+i*myWidth for k in range(len(xlabels))], # one bar for each feature
                       height=tab[measure], # the measure value
                       width=myWidth,
                       color=[dictPalette[col] 
                              if dictPalette 
                              else "r" 
                              for col in xlabels], # if each feature has a specific color, we use it
                       ec="k", # black bar edges 
                       alpha=(i+1)/l
                      )

            axs[j].set_xlim(0,len(xlabels)+1) # set x axis min and max
            axs[j].set_xticks([k+1 for k in range(len(xlabels))]) # set positions of ticks
            axs[j].set_xticklabels(xlabels,rotation=45,ha="right") # features names for ticklabels
            axs[j].set_title("Mesure de performance - "+measure,fontsize=15,fontweight="heavy") # name of measure for axe title
    

    # create a legend if several=1
        if several==1 :
            # we create a bar for each test
            axs[-1].bar(x=xOffset+i*myWidth,
                            height=1,
                            width=myWidth,
                            color="grey",
                            ec="k",
                            alpha=(i+1)/l
                           )
            # we put each test name in its bar
            axs[-1].text(x=xOffset+i*myWidth,
                         y=1,
                         s=testsNamesList[i],
                         fontsize=25/m,
                         color="k",
                         rotation="vertical",
                         ha="center",
                         va="top"
                         )
            axs[-1].set_xlim(0.25,1.75) # set x axis min and max
            axs[-1].set_ylim(0,1.1) # set y axis min and max
            axs[-1].set_xticklabels("") # ne need for labels
            axs[-1].set_yticklabels("")
            axs[-1].set_title("Legend",fontsize=12,style="italic") # put "legend" in title
            axs[-1].set_facecolor((1, 1, 1)) # no need for grid
            axs[-1].spines[["left", "right", "top", "bottom"]].set_color("k") # just need for the "box"

    axs[0].set_ylim(bottom=0,top = max([tab["RMSE"].max() for tab in measuresTabs])*1.1) # set y axis min max for RMSE barplot
    axs[1].set_ylim(bottom=0,top=1) # set y axis min for R2 bar plot
    
    plt.show()


def ShowResultsNumImputsCHOICES (measuresTabs,testsNamesList,dictPalette=None,singleFigsize=(6,5)) :

    '''draw barplots  of numerical features imputation tests results (one for RMSE, one for R2)

    parameters
    ----------
    measuresTabs : list of dataframes, output(s) of the resultsNumImput function
    testNames : list of strings, name(s) of the imputation test(s)

    optionnal
    ---------
    dictPalette : dictionnary, with features names for keys and colors for values
    returns
    -------
    resultTab : a figure with 2 or 3 axes
        - 2 barplots : one for RMSE and one for R2
        - if several tests results, draw a legend on the 3rd axe
    '''
    

    # create ipywidgets dropdowns
    import ipywidgets as widgets
    
    
    widGraph=widgets.RadioButtons(options=["Multi - All tests and measures","Single - One test and one measure"],
                                     value="Single - One test and one measure",
                                     description="",
                                     disabled=False,
                                    style={'description_width': 'initial'}
                                    )
    ui=widgets.HBox([widgets.Label("Plotting option :"),widGraph])
    
    def witchGraphFunc(graphType) :
        if graphType == "Multi - All tests and measures" :
            ShowResultsNumImputsMulti (measuresTabs=measuresTabs,testsNamesList=testsNamesList,dictPalette=dictPalette)
        if graphType == "Single - One test and one measure" :
            ShowResultsNumImputsWID (measuresTabs=measuresTabs,testsNamesList=testsNamesList,dictPalette=dictPalette,figsize=singleFigsize)
    
    out = widgets.interactive_output(witchGraphFunc, {"graphType" : widGraph})
    
    display(ui,out)
    


def makeCorrGroups(df, threshold) :
    '''
    A function to make groups of columns based on correlation of one member to another
    
    parameters
    ----------
    df : dataframe
    threshold : float in [0,1], correlation threshold
    
    returns
    -------
    groupsDict : dictionnary with groups, each with the list of features
    '''
    # compute correlation matrix
    corrMatrix = df.corr()
    
    # Generate a mask to keep only the lower triangle
    maskTri = np.triu(np.ones_like(foodCorr, dtype=bool),k=0)
    
    # apply to the matrix
    corrMatrix = corrMatrix.mask(maskTri)
    
    # create a group if there is at least one value > threshold in a column
    
    # put first member of the group in a list
    groupsFirstMembers = [col for col in corrMatrix.columns if any(np.abs(corrMatrix[col])>=threshold)]
    # initiate a dictionnary
    groupsDict = {i : [groupsFirstMembers[i]] for i in range(len(groupsFirstMembers))}
    # complete each group
    for G in groupsDict.keys() :
        for col in corrMatrix.columns :
            if np.abs(corrMatrix.loc[col,groupsDict[G][0]])>=threshold :
                groupsDict[G].append(col)
    
    # drop group if all feature are already in another one
    keysToDrop = set()
    for Gi in groupsDict.keys() :
        for Gj in groupsDict.keys() :
            if Gi==Gj : # iterate on the other groups
                continue
            if all(col in groupsDict[Gi] for col in groupsDict[Gj]) : # check if all feature of Gj are in Gi
                keysToDrop.add(Gj)
    
    for G in keysToDrop :
        groupsDict.pop(G) # drop selected groups
    
    # rename groups
    groupsDict = {"G"+str(list(groupsDict.keys()).index(k)+1) : G for k,G in groupsDict.items()}
    
    # put not correlated feature in a last group
    groupsDict["others"] = [col for col in corrMatrix.columns if col not in sum(groupsDict.values(),[]) ]
    
    return groupsDict


def featureDistrib(df,featureName,dictPalette=None,ax=None) :
    '''
    A function to draw the empirical distribution of a given feature
    
    parameters
    ----------
    df : dataframe
    
    optionnal
    ---------
    dictPalette : dictionnary, with features names for keys and colors for values. By default None
    ax = axe position, if used within a matplotlib subplots
    

    '''
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # common parameters
    sns.set_theme()
    myColor=dictPalette[featureName] if dictPalette else None # set the color of graph with dictPalette
    myStat="density" # graph will display percentages
    
    if (df[featureName].dtype in ["float64","float32","float16","float8","int64","int32","int16","int8"]) and df[featureName].nunique()!=2 : # for numerical features
        sns.histplot(data=df,
                     x=featureName,
                     color=myColor,
                     kde=True, # show density estimate
                     ax=ax,
                     stat=myStat,
                    ) # draw
    else : # for other categorical features
        myOrder=df[featureName].value_counts().index.tolist() # sort feature categories by number of appearence
        myOrderedCatSeries=pd.Categorical(df[featureName],myOrder) # create a categorical Series with that order
        sns.histplot(y=myOrderedCatSeries,
                     color=myColor,
                     ax=ax,
                     stat=myStat

                   )
    # set title if ax=None
    if not ax :
        plt.title(featureName+" - Distribution empirique");


def compareFeatureDistrib (newDf,oldDf,featureName,dictPalette=None,figsize=(15,6)) :
    '''
    A function to compare the empirical distribution of a given feature, before and after imputations
    
    parameters
    ----------
    newDf : dataframe with imputed feature
    oldDf : dataframe with non-imputed feature
    featureName : name of the feature
    
    optionnal
    ---------
    dictPalette : dictionnary, with features names for keys and colors for values. By default None
    figsize : tuple or list, dimensions of the figure. By default (15,6)
    

    '''
    sns.set_theme()
    fig,axs=plt.subplots(1,2,figsize=figsize,sharex=True,sharey=True) # create a 2 axes subplots
    fig.suptitle("Impact des imputations sur la distribution empirique de "+featureName) # set title
    featureDistrib(oldDf,featureName,dictPalette,ax=axs[0]) # use featureDistrib() on non-imputed version
    featureDistrib(newDf,featureName,dictPalette,ax=axs[1]) # use featureDistrib() on imputed version

    
    # set subtitles
    axs[0].set_title("sans Imputation")
    axs[1].set_title("avec Imputation");
    


def compareDfDistribs (newDf,oldDf,colList,dictPalette=None,figsize=(15,6)) :
    '''
    A function to compare the empirical distribution of any of a given features list, before and after imputations
    
    parameters
    ----------
    newDf : dataframe with imputed feature
    oldDf : dataframe with non-imputed feature
    colList : list of features
    
    optionnal
    ---------
    dictPalette : dictionnary, with features names for keys and colors for values. By default None
    figsize : tuple or list, dimensions of the figure. By default (15,6)
    

    '''
    # 


    import ipywidgets as widgets
    widFeature=widgets.Dropdown(options=colList,
                                              description="Feature :",
                                              disabled=False,
                                              style={'description_width': 'initial'}
                                             )
    
    out = widgets.interactive_output(compareFeatureDistrib, {"newDf" : widgets.fixed(newDf),
                                                             "oldDf" : widgets.fixed(oldDf),
                                                             "featureName" : widFeature ,
                                                             "dictPalette" : widgets.fixed(dictPalette),
                                                             "figsize":widgets.fixed(figsize)
                                                               }
                                    )
    display(widFeature,out)


def distribRidgePlot(df, categFeatureName, numFeatureName,palette=None,overlap=0.5,order=None,zoomInterval=None,clip=None) :
    '''
    A function to compare the empirical distribution of any of a given features list, before and after imputations
    
    parameters
    ----------
    newDf : dataframe with imputed feature
    oldDf : dataframe with non-imputed feature
    colList : list of features
    
    optionnal
    ---------
    palette : dictionnary, with features names for keys and colors for values. By default None
    overlap : float, to adjust space between facets/axes. Default = 0.5
    order : list of labels of the categorical feature, in a given order
    zoomInterval : tuple or list, zoom interval on x axis. By default None
    clip : tuple or list, kdeplot interval
    

    '''
    # set theme
    sns.set_theme(
        style="white", # seaborn white style
        rc={"axes.facecolor": (0, 0, 0, 0), # set facecolor with alpha = 0, so the background is transparent
            'axes.linewidth':2} # thick line width
    )
    
    # config the order of categories on the grid
    if order :
        myRowOrder = order
    else :
        myRowOrder = df.groupby(categFeatureName)[numFeatureName].median().sort_values(ascending=False).index.tolist()
    
    # config palette
    nbLabels=df[categFeatureName].nunique()
    if palette :
        if type(palette)==dict :
            myPalette=palette
        elif type(palette)==str :
            myPalette={lab : c for lab,c in zip(myRowOrder,sns.color_palette(palette,n_colors=nbLabels))}
        else :
            myPalette={lab : c for lab,c in zip(myRowOrder,sns.color_palette(n_colors=nbLabels))}
    else :
        myPalette={lab : c for lab,c in zip(myRowOrder,sns.color_palette(n_colors=nbLabels))}
    
    # initiate a FacetGrid object
    g = sns.FacetGrid(
        data=df, 
        row=categFeatureName, # create one subplot for each category of feature
        hue=categFeatureName, # use a different color plot for each subplot
        palette=myPalette, # use given palette
        row_order=myRowOrder, # order the categories/levels
        height=1.2, # height of each facet
        aspect=9 # height ratio to give the width of each facet
        
    )
    
    # draw density plots
    # ones filled with a color
    g.map_dataframe( # apply plotting function to each facet
        sns.kdeplot, # density function
        x=numFeatureName, # use with our numerical feature
        fill=True, # fill area under plot
        alpha=1,
        clip=clip
    )
    # ones with only a black line
    g.map_dataframe(
        sns.kdeplot,
        x=numFeatureName,
        color="black",
        clip=clip
    )
    
    # set main title
    plt.suptitle("Distibution de '"+numFeatureName+"' en fonction de '"+categFeatureName+"'")
    
    # set facets titles
    # plots will overlap so we need to place titles in an unusual place
    def writeCategory(x, color, label): # create a plotting function to write titles
        ax = plt.gca() # get current axes
        ax.text(x=0, # left of the distribution plot
                y=0.05/overlap, # above x axis line
                s=label, # write current category/label of categFeatureName of the facet
                color=color, # use the current color of the facet
                fontsize=10, # def texte size
                ha="left", va="center", # vertical and horizontal alignments
                transform=ax.transAxes  # use coordinate system of the Axes
               )
    g.map(writeCategory, numFeatureName) # draw our titles on each facet
    g.set_titles("") # get rid of classic facet titles
    
    # make plots overlap
    g.fig.subplots_adjust(hspace=-overlap)

    # set ticks and ticklabels
    g.set(ylabel="", # we do not need the y label (always the same)
          yticks=[], # focus on shape, no need for y ticks
          xlabel=numFeatureName, # set x label
          xlim=zoomInterval # zooming if asked
         ) 
    g.despine( left=True); # get rid of y axis
    


def displayByInterval (
    df, 
    catFeatureName,
    numFeatureName, 
    middle,
    rangeSize,
    colorCatFeatureName=None,
    palette=None,
    autoscale=True,
    plotStyle="bar") :
    
    '''
    display a empirical distribution plot of a categorical variable, for a given range of values of a numerical variable
    
    parameters :
    ------------
    df : dataframe
    catFeatureName : string, name of the categorical variable we want to display the distribution of
    numFeatureName : strint, name of the numerical variable we want to filter on
    middle : float, middle of the filter range
    rangeSize : float, size of the filter range
    
    optional parameters :
    ---------------------
    colorCatFeatureName : string, name of a categorical variable we want to use for subgroups. Default : None
    palette : dict, colors associated with each colorCatFeatureName label. Default : None
    autoscale : bool, wether or not one wants to autoscale the count axis. 
                - If True, axis' max will be the largest number of observations among labels, for filtered dataframe. 
                - If False, axis' max will be the largest number of observations among labels, for the whole dataframe.
                Default : True
    plotStyle : string, name of a plotly express bar chart : "bar" or "bar_polar". Default : "bar"
    
    '''
    
    # filter the dataframe
    mask=(df[numFeatureName]>=middle-rangeSize/2)&(df[numFeatureName]<=middle+rangeSize/2) # create of mask
    filteredDf=df.loc[mask] # filter
    
    # use .groupby() for counting occurrences
    countCol=filteredDf.notna().sum().idxmax() # select a column for counting
        
    if colorCatFeatureName : # if we have a subgroup variable, we use it for .groupby()
        groupByCol=[catFeatureName,colorCatFeatureName]
    else :
        groupByCol=[catFeatureName]
        
    countDf=filteredDf.groupby(groupByCol)[countCol].count().reset_index() # create count dataframe
    
    # handle palette. If not compatible with colorCatFeatureName, don't use it
    if palette :
        if type(palette)!=dict :
            palette=None
        if colorCatFeatureName  :
            for label in df[colorCatFeatureName].unique() :
                if label not in palette.keys() :
                    palette=None
                    break
        else :
            palette=None
            
    # plot 
    import plotly.express as px # import library
    
    myLabels= {countCol : "count"}
    
    if plotStyle=="bar_polar" : 
        fig=px.bar_polar(countDf, 
                         r=countCol, 
                         theta=catFeatureName, 
                         color=colorCatFeatureName, 
                         template="seaborn",
                         color_discrete_map= palette,
                         color_discrete_sequence=None if palette else px.colors.qualitative.Light24_r,
                         range_r= None if autoscale else [0,df[catFeatureName].value_counts().max()],
                         labels=myLabels

                      )
    
    if plotStyle=="bar" : 
        fig=px.bar(countDf, 
                         y=countCol, 
                         x=catFeatureName, 
                         color=colorCatFeatureName, 
                         template="seaborn",
                         color_discrete_map= palette,
                         color_discrete_sequence=None if palette else px.colors.qualitative.Light24_r,
                         range_y= None if autoscale else [0,df[catFeatureName].value_counts().max()],
                         labels=myLabels

                      )
    
    # set title
    fig.update_layout(
    height=800,
    title_text="Distribution de "+catFeatureName+\
        " pour "+numFeatureName+" compris entre "+\
        str(round(middle-rangeSize/2,1))+\
        " et "+str(round(middle+rangeSize/2,1)),
    )
    
    
    
    fig.show("notebook")



def displayByIntervalWID (df,palette=None) :

        
    '''
    runs displayByInterval function, and uses ipywidgets to choose its arguments
    
    parameters :
    ------------
    df : dataframe
        
    optional parameters :
    ---------------------
    palette : dict, colors associated with each colorCatFeatureName label. Default : None
    
    '''
    
    # create widgets
    
    import ipywidgets as widgets # import library
    # create a list to store widgets
    widgetsList=[]
    
    # manage layouts et Styles for widgets
    widLayout={i : widgets.Layout(width="auto",
                               grid_area="a"+str(i)
                              ) for i in range(7)}
    
    widSliderStyle={'description_width': 'initial'}
    widSelectionStyle={'description_width': 'initial'}

    
    
    # widget - numerical variable selection
    numColList=df.select_dtypes('float64').columns.tolist()
    numColList.sort()
    widNumCol=widgets.ToggleButtons(options=numColList, 
                                    value=numColList[0],
                                    description="Numerical variable      :",
                                    disabled=False,
                                    style=widSelectionStyle,
                                    button_style='info',
                                    layout=widLayout[0]
                                    )
    widgetsList.append(widNumCol)
        
    # widget - numerical variable values range size
    widRangeSize=widgets.FloatSlider(value=(df[widNumCol.value].max()-df[widNumCol.value].min())/4,
                                     min=0,
                                     max=df[widNumCol.value].max()-df[widNumCol.value].min(),
                                     step=0.1,
                                     description="Range size for "+widNumCol.value+" :",
                                     disabled=False,
                                     orientation='horizontal',
                                     readout=True,
                                     readout_format='.1f',
                                     style=widSliderStyle,
                                     layout=widLayout[1]
                                    )
    widgetsList.append(widRangeSize)
    
    # widget - numerical variable middle value
    widRangeMiddle=widgets.FloatSlider(value=df[widNumCol.value].min()+widRangeSize.value/2,
                                     min=df[widNumCol.value].min()+widRangeSize.value/2,
                                     max=df[widNumCol.value].max()-widRangeSize.value/2,
                                     step=0.1,
                                     description="Value for "+widNumCol.value+" :",
                                     disabled=False,
                                     orientation='horizontal',
                                     readout=True,
                                     readout_format='.1f',
                                     style=widSliderStyle,
                                     layout=widLayout[2]
                                     )
    widgetsList.append(widRangeMiddle)
        
    def handle_Num_dropdown_change(change): # linking sliders with the name of the numerical variable
        widRangeSize.value=(df[change.new].max()-df[change.new].min())/4
        widRangeSize.max=df[change.new].max()-df[change.new].min()
        widRangeSize.description="Range size for "+change.new+" :"
        
        widRangeMiddle.value=df[change.new].min()+widRangeSize.value/2
        widRangeMiddle.min=df[change.new].min()+widRangeSize.value/2
        widRangeMiddle.max=df[change.new].max()-widRangeSize.value/2
        widRangeMiddle.description="Value for "+change.new+" :"
        
    widNumCol.observe(handle_Num_dropdown_change,'value')
    
    
    def handle_keep_my_range(change): # linking the middle value slider maximum and minimum with the size of the range
        widRangeMiddle.min=df[widNumCol.value].min()+change.new/2
        widRangeMiddle.max=df[widNumCol.value].max()-change.new/2

    widRangeSize.observe(handle_keep_my_range,'value')
    
    # widget - categorical variable selection 
    catColOptions = df.select_dtypes("category").columns.tolist()
    widCat = widgets.ToggleButtons(options=catColOptions, 
                                    value=catColOptions[0],
                                    description="Categorical variable :",
                                    disabled=False,
                                    style=widSelectionStyle,
                                    button_style='info',
                                    layout=widLayout[3]
                                    )
    widgetsList.append(widCat)
    
    # widget - hue/color categorical variable selection  
    ColorCatColOptions = [None]+df.select_dtypes("category").columns.tolist()
    ColorCatColOptions.remove(widCat.value)
    widColorCat = widgets.ToggleButtons(options=ColorCatColOptions, 
                                    value=None,
                                    description="Feature for Color :",
                                    disabled=False,
                                    style=widSelectionStyle,
                                    button_style='info',
                                    layout=widLayout[4]
                                    )
    widgetsList.append(widColorCat)
    
    def handle_Cat_dropdown_change_1(change): # linking both categorical dropdowns, variables names available
#         if change.new==widColorCat.value :
#             widColorCat.value=None
        widColorCat.options=[op for op in widColorCat.options if op!=change.new]+[change.old]
        
    widCat.observe(handle_Cat_dropdown_change_1,'value')
    
    # widget - autoscale of the radius axis, yes or no
    widPlotStyle=widgets.ToggleButtons(options=["bar","bar_polar"],
                                          value="bar",
                                          description="Plot style :",
                                          disabled=False,
                                          style=widSelectionStyle,
                                          button_style='info',
                                          layout=widLayout[5]
                                          )
    widgetsList.append(widPlotStyle)
    
    widAutoscale=widgets.ToggleButtons(options={"yes" : True , "no" : False},
                                          value=True,
                                          description="Radius autoscale :",
                                          disabled=False,
                                          style=widSelectionStyle,
                                          button_style='info',
                                          layout=widLayout[6]
                                          )
    widgetsList.append(widAutoscale)
    
    # graph
    out = widgets.interactive_output(displayByInterval, {"df" : widgets.fixed(df),
                                                         "numFeatureName" : widNumCol ,
                                                         "catFeatureName" : widCat,
                                                         "middle" : widRangeMiddle,
                                                         "rangeSize" : widRangeSize,
                                                         "colorCatFeatureName" : widColorCat,
                                                         "palette" : widgets.fixed(palette),
                                                         "autoscale" : widAutoscale,
                                                         "plotStyle" : widPlotStyle

                                                    }
                                    )

    # create ui

    # manage layout for ui
    gridLayout = widgets.Layout(width='100%',
                                grid_template_rows='auto auto auto',
                                grid_template_columns='16% 16% auto auto 16% 16%',
#                                 grid_gap="0% 0%",
                                grid_template_areas='''
                                "a0 a0 a3 a3 a4 a4"
                                "a1 a1 a2 a2 a2 a2"
                                "a5 a5 a5 a6 a6 a6"
                                
                                '''
                               )
    
    ui = widgets.GridBox(children=widgetsList,
                 layout=gridLayout)
    

    
    #display
    display(ui,out)
    

def myOneWayAnova (df,categFeatureName,numFeatureName,alpha) :
    '''
    return a modified version of the generic One Way ANOVA table from statsmodels.api
    
    inputs
    -----
    df : dataframe
    categFeatureName : string, name of the categorical variable
    numFeatureName : string, name of the numerical variable
    alpha : float, significance level
    
    
    outputs
    ------
    model : statsmodel ols object, our anova
    anovaTable : dataframe with 
        - 3 rows :
            - Model (from the labels of the categorical feature)
            - Residual_Error
            - Total
        - 5 columns :
            - Sum_of_Squares
            - Degrees_Freedom
            - Mean_Square : SumSquares/degFreed
            - F_statistic : MeanSquareModel/MeanSquareResidual
            - Critical_F : critical F_value for these degrees of freedom
            - F_test_p_value : probability of getting this F-statistic (or a larger one) given H0
            - Eta_Square : SumSquaresModel/TotalSumSquares
    
    '''
    # import libraries
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    import scipy.stats as stats
    
    # ANOVA using ols
    formula=numFeatureName+"~"+"C("+categFeatureName+")"
    model=ols(formula,data=df).fit()
    
    # classic statsmodels ANOVA table
    anovaTable = sm.stats.anova_lm(model,typ=2).reset_index()
    anovaTable.pop("index")
    
    # rename columns
    anovaTable.rename(
        index={0:"Model",1:"Residual_Error"},
        columns={"sum_sq":"Sum_of_Squares","df":"Degrees_Freedom","F":"F_statistic","PR(>F)":"F_test_p_value"},
        inplace=True
    )
    
    # add mean square column
    anovaTable["Mean_Square"]=anovaTable["Sum_of_Squares"]/anovaTable["Degrees_Freedom"]
    # add total row
    anovaTable.loc["Total",["Sum_of_Squares","Degrees_Freedom"]]=anovaTable[["Sum_of_Squares","Degrees_Freedom"]].sum(axis=0)
    # add eta squared value
    anovaTable.loc["Model","Eta_Square"]=anovaTable.loc["Model","Sum_of_Squares"]/anovaTable.loc["Total","Sum_of_Squares"]
    # critical F statistic value
    
    anovaTable.loc["Model","Critical_F"]=stats.f.ppf( # find the F-value for a given percentage of area under a F-distribution
        q=1-alpha, # percentage
        dfn=anovaTable.loc["Model","Degrees_Freedom"], # numerator degrees of freedom of the F-distribution
        dfd=anovaTable.loc["Residual_Error","Degrees_Freedom"] # denominator degrees of freedom
    )
    
    # change order of columns
    anovaTable=anovaTable[["Sum_of_Squares","Degrees_Freedom","Mean_Square","F_statistic","Critical_F","F_test_p_value","Eta_Square"]]
    
    return model, anovaTable


def myPCA (df, q, ACPfeatures=None) :
    '''
    run scaling preprocessing, using sklearn.preprocessing.StandardScaler, 
    and PCA, using scikit learn sklearn.decomposition.PCA 
    
    parameters :
    ------------
    df : DataFrame on which we want to run the PCA
    q : number of components of the PCA
    
    optionnal parameters :
    ----------------------
    ACPfeatures = list of columns names of df used for PCA. By default None (in that case : all dtype 'float64' columns names)
    
    outputs :
    ---------
    X_scaled : values of scaled df
    Xidx : index of df
    Xfeatures : columns names
    dfPCA : PCA fitted with df
    
        
    '''
    # imports
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    
    # stores values and index 
    Xfeatures = ACPfeatures if ACPfeatures else [col for col in df.columns.tolist() if df[col].dtype.kind in 'biufc']
    PCAdf=df.copy()[Xfeatures]
    X = PCAdf.values
    Xidx=PCAdf.index
    
    
    # scale
    scaler = StandardScaler() # instantiate
    X_scaled = scaler.fit_transform(X) # fit transform X
    
    # PCA
    dfPCA = PCA(n_components = q)
    dfPCA.fit(X_scaled)
    
    return X_scaled, Xidx, Xfeatures, dfPCA



def PCA_scree_plot (pca) :
    '''
    draw the eigen values scree plot of a given fitted pca
    
    parameter : fitted sklearn.decomposition.PCA
    '''
    import pandas as pd
    import numpy as np
    # initiate dataframe
    screeDf=pd.DataFrame(index=["F"+str(k+1) for k in range(pca.n_components)])
    # explained variance ratio in percentage
    screeDf["Variance expliquée"] = (pca.explained_variance_ratio_).round(2)
    # cumsum
    screeDf["Variance expliquée cumulée"] = screeDf["Variance expliquée"].cumsum().round(2)
    
    # plot
    import seaborn as sns
    sns.set_theme()

    import plotly.express as px
    import plotly.graph_objects as go
    
    fig=px.bar(screeDf,y="Variance expliquée",text_auto=True)
    fig2=go.Scatter(y=screeDf["Variance expliquée cumulée"],x=screeDf.index,mode="lines+markers",showlegend=False,name="")
    fig.add_trace(fig2)

    fig.layout.yaxis.tickformat = ',.0%'
    
    for idx,val in screeDf["Variance expliquée cumulée"].iloc[1:].items() :
        fig.add_annotation(y=val,x=idx,text=str(round(val*100))+"%",showarrow=False,yshift=10,xshift=-10)
    
    fig.update_layout(
    height=800,
    title_text="Eboulis des valeurs propres",
    xaxis_title="Composante principale",
    yaxis_title="Valeurs propres - variance expliquée",
)
    
    fig.show()



def pcaCorrelationMatrix (pca,
                          PcafeaturesNames,
                          additionnalVariable=None
                          ) :
    '''
    return the correlation matrix 'features <-> loadings', dataframe 
    Parameters :
    -----------
    pca : sklearn.decomposition.PCA : PCA object, already fitted
    PcafeaturesNames : list or tuple : list of pca features names
    
    Optional parameters :
    ---------------------
    additionnalVariable : list or tuple containing elements to add another variable to the matrix (i.e. another row)
        - element 0 : X_scaled, used for PCA
        - element 1 : addVarSeries, the additionnal variable pandas.Series object
    '''
    import pandas as pd
    import numpy as np
    matrix=pca.components_.T*np.sqrt(pca.explained_variance_)
    dfMatrix=pd.DataFrame(
        matrix,
        index=PcafeaturesNames,
        columns=[
            "F"+str(i+1)+" ("+str(round(100*pca.explained_variance_ratio_[i],1))+"%)" for i in range(pca.n_components_)
        ]
    )
    
    if additionnalVariable :
        X_scaled=additionnalVariable[0]
        addVarSeries=additionnalVariable[1]

        
        C = pd.DataFrame(pca.transform(X_scaled),index=addVarSeries.index,columns=dfMatrix.columns)
        corrAddVar=C.corrwith(addVarSeries,axis=0)        
        
        dfMatrix.loc["Add Var ("+addVarSeries.name+")"]=corrAddVar
    return dfMatrix



def heatPcaCorrelationMatrix (pca,
                              PcafeaturesNames,
                              additionnalVariable=None,
                              figsize=(10,5)) :
    '''
    display a PCA correlation matrix in a Seaborn Heatmap way
    
    parameters :
    ----------
    pca : sklearn.decomposition.PCA : PCA object, already fitted
    PcafeaturesNames : list or tuple : list of pca features names
    
    
    optional parameters :
    --------------------
    additionnalVariable : list or tuple containing elements to add another variable to the matrix (i.e. another row)
        - element 0 : X_scaled, used for PCA
        - element 1 : addVarSeries, the additionnal variable pandas.Series object
    figsize : list or tuple, size of the figure. Default = (10,5)
    '''
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # use pcaCorrelationMatrix function to create the matrix
    dfMatrix=pcaCorrelationMatrix (pca=pca,
                                   PcafeaturesNames=PcafeaturesNames,
                                   additionnalVariable=additionnalVariable
                                  )
    
    # initiate plot
    fig,ax=plt.subplots(1,figsize=figsize)
    sns.set_theme()
    
    # heatmap
    sns.heatmap(
        data=dfMatrix, # use the correlation matrix computed above
        linewidth=1, # line between squares of the heatmap
        cmap=sns.diverging_palette(262,12, as_cmap=True,center="light",n=9), # blue for anticorrelated, red for correlated
        center=0, # no color for no correlation
        annot=True, # displays Pearson coefficients
        fmt="0.2f", # with 2 decimals
        ax=ax
    );
    
    # change tick labels locations
    ax.tick_params(
        top=False,
        labeltop=True, # put them on top
        labelbottom=False,
        bottom=False
    )
    
    # set title
    ax.set_title("Corrélations variables / composantes principales")


def heatPcaCorrelationMatrixWid (df,
                                 X_scaled,
                                 pca,  
                                 PcafeaturesNames,
                                 figsize=(10,5)
                                ) :
    '''
    display a PCA correlation matrix in a Seaborn Heatmap way
    
    parameters :
    ----------
    pcaCorrelationMatrix : dataframe, PCA correlation matrix returned from the pcaCorrelationMatrix function
    
    optional parameters :
    --------------------
    figsize : list or tuple, size of the figure. Default = (10,5)
    '''

    import ipywidgets as widgets
    # create a widget for choosing additionnal variable
    addVarColList=[col for col in df.columns if (col not in PcafeaturesNames)and(df[col].dtype.kind in 'biufc')]

    import ipywidgets as widgets
    widAddVar=widgets.Dropdown(options={col : (X_scaled,df[col]) for col in addVarColList}|{None : None},
                               value=None,
                               description="Additionnal variable :",
                               disabled=False,
                               style={'description_width': 'initial'}
                                    )    
    

    out=widgets.interactive_output(heatPcaCorrelationMatrix, {"pca" : widgets.fixed(pca),
                                                              "PcafeaturesNames" : widgets.fixed(PcafeaturesNames),
                                                              "additionnalVariable":widAddVar,
                                                              "figsize" : widgets.fixed(figsize)
                                                             }
                                  )
    
    display(widAddVar,out)



def correlation_graph_enhanced(pca, 
                               x_y, 
                               PcafeaturesNames,
                               normalization,
                               dictPalette=None,
                               figsize=(10,9),
                               additionnalVariable=None
                              ) : 
    """display correlation graph for a given pca

    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    Parameters : 
    ----------
    pca : sklearn.decomposition.PCA : PCA object, already fitted
    x_y : list or tuple : the couple of the factorial plan, example [1,2] for F1, F2
    PcafeaturesNames : list or tuple : list of pca features names
    normalization : string, decide what one wants to plot :
    - "loadings" - columns of V.Lambda^(1/2) - loadings
    - "principal_axis" - columns of V - principal directions/axis
    
    Optional parameters :
    -------------------
    dictPalette : dictionnary, with features names for keys and colors for values - default : None
    figsize : list or tuple, size of figure - default : (10,9)
    additionnalVariable : list or tuple containing elements to add another variable to the graph
        - element 0 : X_scaled, used for PCA
        - element 1 : addVarSeries, the additionnal variable pandas.Series 
    """
    sns.set_theme()
    # Extract x and y 
    x,y=x_y
    
  
    # Adjust x and y to list/array numerotation
    x=x-1
    y=y-1

    
    # compute matrix
    if normalization=="principal_axis" : 
        matrix=pca.components_.T # principal axis matrix
        dfMatrix = pd.DataFrame( # tranform matrix into a dataframe
            matrix,
            index=PcafeaturesNames,
            columns=[ # one columns for each principal vector, with its global quality of representation
                # which is the percentage of explained variance 
            "F"+str(i+1)+" ("+str(round(100*foodPCA.explained_variance_ratio_[i],1))+"%)" for i in range(foodPCA.n_components_)
            ]
        )
    if normalization=="loadings" : # in that we use the above function
        dfMatrix=pcaCorrelationMatrix(pca=pca,
                                       PcafeaturesNames=PcafeaturesNames,
                                       additionnalVariable=additionnalVariable)
    
    # size of the image (in inches)
    fig, ax = plt.subplots(figsize=figsize)
    
    # For each column :
    for i,col in enumerate(dfMatrix.index.tolist()) :
        x_coord = dfMatrix.iloc[i,x]
        y_coord = dfMatrix.iloc[i,y]
        ax.arrow(0,
                 0, # Start the arrow at the origin 
                x_coord,  
                y_coord,  
                head_width=0.07,
                head_length=0.07, 
                width=0.02, 
                length_includes_head=True, # so arrow stays inside R=1 circle
                color=dictPalette[col if col in dictPalette.keys() else additionnalVariable[1].name] , # if each feature has a specific color, we use it
                )

        
        # put name of the feature at the top of the arrow
       
        ax.text(x_coord,
                y_coord,
                col,
                horizontalalignment="left" if x_coord>0 else "right",
                verticalalignment="bottom" if y_coord>0 else "top",
                fontsize=10*figsize[0]/10,
                rotation=np.arctan(y_coord/x_coord)*180/np.pi
                )
    

        
    # Display x-axis and and y-axis in dot ligns
    plt.plot([-1, 1], [0, 0], color='grey', ls='--')
    plt.plot([0, 0], [-1, 1], color='grey', ls='--')

    # Names of factorial axis, with percent of explained variance/inertia
    plt.xlabel(dfMatrix.columns.tolist()[x],fontsize=12*figsize[0]/10)
    plt.ylabel(dfMatrix.columns.tolist()[y],fontsize=12*figsize[0]/10)
 
    # ticks size
    plt.setp(ax.get_xticklabels(),fontsize=13*figsize[0]/10)
    plt.setp(ax.get_yticklabels(),fontsize=13*figsize[0]/10)

    plt.title("Cercle des corrélations (F{} et F{})".format(x+1, y+1),fontsize=13*figsize[0]/10)
    


    # circle if we use loadings
    if normalization=="loadings" :
        an = np.linspace(0, 2 * np.pi, 100)
        plt.plot(np.cos(an), np.sin(an),c="k")  # Add a unit circle for scale

    # Axes and display
    plt.axis('equal') #common scale
    plt.show(block=False)


def correlation_graph_enhanced_WID(df,
                                   X_scaled,
                                   pca,  
                                   PcafeaturesNames,
                                   dictPalette=None,
                                   figsize=(10,9),
                                  ) : 
    """display correlation graph for a given pca, with the choice of normalization


    Parameters : 
    ----------
    pca : sklearn.decomposition.PCA : PCA object, already fitted
    x_y : list or tuple : the couple of the factorial plan, example [1,2] for F1, F2
    PcafeaturesNames : list or tuple : list of features (ie dimensions) to draw
    
    Optional parameters :
    -------------------
    dictPalette : dictionnary, with features names for keys and colors for values - default : None
    
    """
    # create ipywidgets radio buttons for normalization choice
    import ipywidgets as widgets
    widNormalization=widgets.RadioButtons(options=["loadings","principal_axis"],
                                          value="loadings",
                                          description="Normalization option :",
                                          disabled=False,
                                          style={'description_width': 'initial'}
                                         )
    # create an ipywidgets dropdown for factorial plan choice
    q=pca.n_components_ # store the numbers of components
    from itertools import combinations # import compbinations tool
    factPlanList=list(combinations([f+1 for f in range(q)],2)) # create a list of all factorial plans
    
    def myOrder(elem) : # create a function to sort the list of factorial plans  
        return(elem[1]-elem[0])
    factPlanList.sort(key=myOrder)
    
    factPlanList=[(str(plan),plan) for plan in factPlanList] #  change format for widgets compatibility
    
    widFactorialPlan=widgets.Dropdown(options=factPlanList,
                                     value=factPlanList[0][1],
                                     description="Factorial plan :",
                                     disabled=False,
                                    style={'description_width': 'initial'}
                                    )
    
    # create a widget for choosing additionnal variable
    addVarColList=[col for col in df.select_dtypes("float64").columns if col not in PcafeaturesNames]
    widAddVar=widgets.Dropdown(options={col : (X_scaled,df[col]) for col in addVarColList}|{None : None},
                               value=None,
                               description="Additionnal variable :",
                               disabled=False,
                               style={'description_width': 'initial'}
                                    )
    
    ui=widgets.HBox([widFactorialPlan,widNormalization,widAddVar])
    
    out = widgets.interactive_output(correlation_graph_enhanced, {"pca" : widgets.fixed(pca),
                                                                  "x_y" : widFactorialPlan ,
                                                                  "PcafeaturesNames" : widgets.fixed(PcafeaturesNames),
                                                                  "normalization" : widNormalization,
                                                                  "dictPalette" : widgets.fixed(dictPalette),
                                                                  "figsize":widgets.fixed(figsize),
                                                                  "additionnalVariable" : widAddVar
                                                               }
                                    )
    
    display(ui,out)




def testDtypes (testCol,testDf) :

    import sys
    import pandas as pd
    df=testDf.copy()
    test=df[testCol]
    
    print(testCol)
    print("dtype : ",test.dtype)
    print("NaN rate : ",test.isna().mean())
    
    if test.dtype != "object" and test.dtype != "category" : 
        if test.min()<0 and test.max()>0 :
            print("<0 and >0")
        elif test.max()<0 :
            print("<0")
        else :
            print(">0")
        print("---------")
        print("raw : ",sys.getsizeof(test)," bytes (with ",test.dtype,")")
        
        l={typ : pd.to_numeric(test,downcast=typ) for typ in ['float','integer','signed','unsigned']}
        
        if test.dtype=="int64" :
            print("integer : ",sys.getsizeof(l['integer'])," bytes (with ",l['integer'].dtype,")")
        print("signed : ",sys.getsizeof(l['signed'])," bytes (with ",l['signed'].dtype,")")
        print("unsigned : ",sys.getsizeof(l['unsigned'])," bytes (with ",l['unsigned'].dtype,")")
        print("float : ",sys.getsizeof(l["float"])," bytes (with ",l["float"].dtype,")")
        
        del l
        
    else :
        print("---------")
        print("raw : ",sys.getsizeof(test)," bytes (with ",test.dtype," )")
        print("Cat : ",sys.getsizeof(test.astype("category"))," bytes (with category)")
    
    del df,test



def bestDtype (series) :
    '''
    returns the most memory efficient dtype for a given Series
    
    parameters :
    ------------
    series : series from a dataframe
    
    returns :
    ---------
    bestDtype : dtype
    '''

    import sys
    import pandas as pd
    import gc
    s=series.copy()

    bestDtype = s.dtype
    bestMemory = sys.getsizeof(s)

    if s.dtype.kind == "O" :
        bestDtype = 'category'
    else :
        for typ in ['unsigned','signed','float'] :
            sDC = pd.to_numeric(s,downcast=typ)
            mem = sys.getsizeof(sDC)
            if mem < bestMemory :
                bestMemory = mem
                bestDtype = sDC.dtype
            del sDC
            gc.collect()
    
    del s
    gc.collect()
    return bestDtype


def plotCatFeatureVsTarget(df, 
                           catFeatureName, 
                           targetFeatureName, 
                           targetValues=None, 
                           includeCatFeatureNan=True, 
                           includeTargetNan=False,
                          ) : 
    '''
    plot the distribution of a categorical feature AND the percentage of target values per category on another graph.
    
    parameters :
    ------------
    df - DataFrame 
    catFeatureName - string : name of the categorical feature 
    targetFeatureName - string : name of the target
    targetValues - list or str/int/float or None : target value(s) to consider in the "percentage" graph. 
                            By default : None  (use of all Target unique values)
    includeCatFeatureNan - bool : Whether or not to include the categorical feature missing values as a category. 
                            By default : True
    includeTargetNan : bool : Whether or not to include the Target missing values as a category. 
                            By default : False,
    
    '''
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import seaborn as sns
    
    # create a copy 
    tempDf = df[[catFeatureName,targetFeatureName]].copy()
    
    # initiate a dataframe to store counts and percentages
    catFeatureDf = pd.DataFrame(tempDf[catFeatureName].value_counts(dropna=not includeCatFeatureNan))
    
    # add a line with infos of the whole dataframe
    catFeatureDf.loc["WHOLE_DATA","count"]=tempDf.shape[0]
    catFeatureDf=catFeatureDf.loc[[catFeatureDf.index[-1]]+list(catFeatureDf.index)[:-1]]
    

    # handle targetValues
    if targetValues :
        if type(targetValues)!=list :
            targetValues = [targetValues]
    else :
        targetValues = list(tempDf[targetFeatureName].dropna().unique())
        if tempDf[targetFeatureName].dtype.kind in "biufc" :
            targetValues.sort()
            
        if includeTargetNan==True and tempDf[targetFeatureName].isna().sum()>0 :
            if tempDf[targetFeatureName].dtype.kind=='O' :
                tempDf[targetFeatureName]=tempDf[targetFeatureName].astype("O").fillna("targetMissing").astype('category')
            else :
                tempDf[targetFeatureName]=tempDf[targetFeatureName].fillna("targetMissing")
            targetValues = targetValues+["targetMissing"]

    
    if tempDf[targetFeatureName].dtype.kind in "biufc" :
        targetValues.sort() 
    
    # add percentages for each Target unique values
    for val in targetValues :
        catFeatureDf[val] = tempDf.loc[tempDf[targetFeatureName] == val,catFeatureName].value_counts(dropna= not includeCatFeatureNan) \
        / catFeatureDf["count"]
            
        catFeatureDf.loc["WHOLE_DATA",val]=tempDf[tempDf[targetFeatureName] == val].shape[0]/df.shape[0]

            
    catFeatureDf=catFeatureDf.reset_index()
    catFeatureDf[catFeatureName]=catFeatureDf[catFeatureName].fillna("'NaN'") # replace np.nan category with a string 'NaN'
    catFeatureDf[catFeatureName]=catFeatureDf[catFeatureName].astype(str)
    catFeatureDf=catFeatureDf.fillna(0) # if a catfeature value was not in filtered tempDf value_counts, i.e. not present
    # in this filtered tempDf, its percentage has been filled with np.nan. Replace with 0% 
    
    # use .melt() method for the "percentage" graph
    catFeatureDfMelt=catFeatureDf.drop(columns="count").melt(id_vars=catFeatureName,
                                                                value_vars=catFeatureDf.columns[2:],
                                                                var_name=targetFeatureName
                                                               )
    
    if len(catFeatureDf) < 10 :
    
        # create figure with 2 axes
        palette=sns.color_palette("Paired",len(catFeatureDf)) # palette
        fig, axs = plt.subplots(1,2,figsize=(12,6)) # figure and axes
        axs = axs.ravel()
        nanRate=str(round(tempDf[catFeatureName].isna().mean()*100,1))+"%"
        fig.suptitle(catFeatureName+" ( "+nanRate+" NaN ) : distribution and relation with Target") # main title

        # plot distribution of the categorical feature
        sns.barplot(data=catFeatureDf.iloc[1:,:],
                    x=catFeatureName,
                    y="count",
                    color="black" if len(targetValues)>1 else None, # if use of several target values, countplot in black
                    palette=None if len(targetValues)>1 else palette, # if use of a unique() target value, use palette colors
                    ax=axs[0])  

        # set axe parameters
        axs[0].set_ylabel("Number of observations")
        axs[0].set_xticklabels(axs[0].get_xticklabels(),rotation=90 if len(catFeatureDf)>3 else 0,ha="center",va="center_baseline")
        axs[0].set_xlabel("")

        # plot percentage of Target values for each category of cat feature
        sns.barplot(data=catFeatureDfMelt,
                    x=catFeatureName,
                    y="value",
                    hue=targetFeatureName if len(targetValues)>1 else None, # if use of several target values, stacked plot
                    ax=axs[1],
                    order=None,
                    palette=sns.color_palette("Greys",len(targetValues)+2)[2:] if len(targetValues)>1 \
                    # if several target values : grey palette
                            else ["black"]+palette # if unique target value : use palette and "black" for "WHOLE_DATA"
                   )

        # set axe parameters
        axs[1].set_ylabel("Percent of "+targetFeatureName+" values" if len(targetValues)>1 \
                          else "Percent of "+targetFeatureName+" = "+str(targetValues[0]))
        axs[1].set_xticklabels(axs[1].get_xticklabels(),rotation=90 if len(catFeatureDf)>3 else 0,ha="center",va="center_baseline")
        axs[1].set_xlabel("")
        axs[1].set_yticks(list(axs[1].get_yticks())+list(catFeatureDf.iloc[0,2:]))
        axs[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        # plot horizontal line(s) for each target value. To compare "WHOLE_DATA" percentage to each category percentage
        for i,val in enumerate(targetValues) :
            axs[1].axhline(y=catFeatureDf.iloc[0,2+i],color=sns.color_palette("Greys",len(targetValues)+2)[2:][i] if len(targetValues)>1 else "black")

            
    else :
        # create figure with 2 axes
        palette=sns.color_palette("Paired",len(catFeatureDf)) # palette
        fig, axs = plt.subplots(1,2,figsize=(12,12)) # figure and axes
        axs = axs.ravel()
        nanRate=str(round(tempDf[catFeatureName].isna().mean()*100,1))+"%"
        fig.suptitle(catFeatureName+" ( "+nanRate+" NaN ) : distribution and relation with Target") # main title
    
        # plot distribution of the categorical feature
        sns.barplot(data=catFeatureDf.iloc[1:,:],
                    y=catFeatureName,
                    x="count",
                    color="black" if len(targetValues)>1 else None, # if use of several target values, countplot in black
                    palette=None if len(targetValues)>1 else palette, # if use of a unique() target value, use palette colors
                    ax=axs[0])  

        # set axe parameters
        axs[0].set_xlabel("Number of observations")
        axs[0].set_ylabel("")
        axs[0].tick_params( # change tick labels locations
            top=True,
            labeltop=True, # put them on top
            labelbottom=True,
            bottom=True
        )

        # plot percentage of Target values for each category of cat feature
        sns.barplot(data=catFeatureDfMelt.loc[catFeatureDfMelt[catFeatureName]!="WHOLE_DATA"],
                    y=catFeatureName,
                    x="value",
                    hue=targetFeatureName if len(targetValues)>1 else None, # if use of several target values, stacked plot
                    ax=axs[1],
                    order=None,
                    palette=sns.color_palette("Greys",len(targetValues)+2)[2:] if len(targetValues)>1 \
                    # if several target values : grey palette
                            else palette # if unique target value : use palette and "black" for "WHOLE_DATA"
                   )

        # set axe parameters
        axs[1].set_xlabel("Percent of "+targetFeatureName+" values" if len(targetValues)>1 \
                          else "Percent of "+targetFeatureName+" = "+str(targetValues[0]))
        axs[1].set_ylabel("")
        axs[1].set_yticklabels("")
        axs[1].set_xticks(list(axs[1].get_xticks())+list(catFeatureDf.iloc[0,2:]))
        axs[1].xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        axs[1].tick_params( # change tick labels locations
            top=True,
            labeltop=True, # put them on top
            labelbottom=True,
            bottom=True
        )

        # plot horizontal line(s) for each target value. To compare "WHOLE_DATA" percentage to each category percentage
        for i,val in enumerate(targetValues) :
            axs[1].axvline(x=catFeatureDf.iloc[0,2+i],color=sns.color_palette("Greys",len(targetValues)+2)[2:][i] if len(targetValues)>1 else "black")

            
        
    del tempDf, catFeatureDf, palette, targetValues, catFeatureDfMelt, nanRate




def plotNumFeatureVsTarget(df, 
                           numFeatureName, 
                           targetFeatureName, 
                           targetValues=None, 
                           includeTargetNan=False,
                           sampleSize=None
                          ) : 
    '''
    plot the distribution of a numerical feature for the whole data AND for filtered data on values of the target.
    
    parameters :
    ------------
    df - DataFrame 
    numFeatureName - string : name of the numerical feature 
    targetFeatureName - string : name of the target
    targetValues - list or str/int/float or None : target value(s) to consider in the "percentage" graph. 
                            By default : None  (use of all Target unique values)
    includeTargetNan : bool : Whether or not to include the Target missing values as a category. 
                            By default : False,
    sampleSize - int or None : size of the df sample ( for faster plotting)
                            By default : None  (no sampling)
    
    '''
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # create a copy 
    if sampleSize :
        tempDf = df[[numFeatureName,targetFeatureName]].copy().sample(sampleSize)
    else :
        tempDf = df[[numFeatureName,targetFeatureName]].copy()
    
    # handle targetValues
    if targetValues :
        if type(targetValues)!=list :
            targetValues = [targetValues]
    else :
        targetValues = list(tempDf[targetFeatureName].dropna().unique())
        if tempDf[targetFeatureName].dtype.kind in "biufc" :
            targetValues.sort()
            
        if includeTargetNan==True :
            if tempDf[targetFeatureName].dtype.kind=='O' :
                tempDf[targetFeatureName]=tempDf[targetFeatureName].astype("O").fillna("targetMissing").astype('category')
            else :
                tempDf[targetFeatureName]=tempDf[targetFeatureName].fillna("targetMissing")
            targetValues = targetValues+["targetMissing"]

    
    
    # filtered df on target values 
    filteredDf = []
    for val in targetValues :
        if type(val)==float and np.isnan(val) : # handle NaN
            filteredDf.append(tempDf.loc[tempDf[targetFeatureName].isna()])
        else :
            filteredDf.append(tempDf.loc[tempDf[targetFeatureName]==val])
    
    # plot
    
    # create figure with 2 axes
    palette=sns.color_palette("Paired") # palette
    fig, axs = plt.subplots(1,2,figsize=(12,6)) # figure and axes
    axs = axs.ravel()
    nanRate=str(round(tempDf[numFeatureName].isna().mean()*100,1))+"%"
    fig.suptitle(numFeatureName+" ( "+nanRate+" NaN ) : distribution and relation with Target") # main title
    
    sns.histplot(data=tempDf, x=numFeatureName,stat='density', bins=100, kde = True, ax=axs[0],ec=None)
    
    for val,df in zip(targetValues,filteredDf) :
        sns.kdeplot(data=df, x=numFeatureName,ax=axs[1],label = targetFeatureName+" = "+str(val),legend=True, cut=0,fill=True)
    
    axs[1].legend()
    

        
    del tempDf, filteredDf, palette, targetValues,nanRate



def plotFeatureVsTargetWID (df,
                               targetFeatureName,
                               targetValsForCat=None,
                                targetValsForNum=None,
                               includeCatFeatureNan=True, 
                               includeTargetNan=False,) :
    import ipywidgets as widgets
    import pandas as pd
    import numpy as np
    
    # widget - categorical variable selection
    colList=[col for col in df.columns if col != targetFeatureName]
    

    widCol=widgets.Dropdown(options=colList,
                              value=colList[0],
                              description="Which column :",
                              disabled=False,
                              style={'description_width': 'initial'}
                              )
    
    widCatOrNumThreshold = widgets.IntSlider(value=50,min=10,max=100,step=10,
                                             description="Nb of unique values below which a num' feature is considered cat' :",
                                            style={'description_width': 'initial'},
                                             disabled=False,
                                             layout=widgets.Layout(width="75%")
                                            )
    
    def whichPlot(col,threshold) :
        if (df[col].dtype.kind in "O") or (len(df[col].unique())<=threshold) :
            
            plotCatFeatureVsTarget(df=df, 
                           catFeatureName=col, 
                           targetFeatureName=targetFeatureName, 
                           targetValues=targetValsForCat, 
                           includeCatFeatureNan=includeCatFeatureNan, 
                           includeTargetNan=includeTargetNan,
                          ) 
        else :
            widSampleSize = widgets.BoundedIntText(value=10000,
                                                   min = 0,
                                                   max = len(df),
                                                   step = 1,
                                                   description='Sample size : ',
                                                   disabled=False
            )
            
            outSub = widgets.interactive_output(plotNumFeatureVsTarget, {"df" : widgets.fixed(df), 
                                                               "numFeatureName" : widgets.fixed(col), 
                                                               "targetFeatureName" : widgets.fixed(targetFeatureName), 
                                                               "targetValues" : widgets.fixed(targetValsForNum), 
                                                               "includeTargetNan" : widgets.fixed(includeTargetNan),
                                                               "sampleSize" : widSampleSize
                                                                        })
            display(widSampleSize,outSub)

    
    out = widgets.interactive_output(whichPlot, {"col" : widCol,
                                                 "threshold" : widCatOrNumThreshold
                                                })
    display(widCol,widCatOrNumThreshold,out)








def extractTNFPFNTP (thresholds, y_pred_proba, y_true) :
    
    '''
    extract lists of TN, FP, FN, TP values for a given list of thresholds
    parameters :
    ------------
    thresholds - list : thresholds
    y_pred_proba - list : prediction scores
    y_true - list : true values
    
    return :
    --------
    TN,FP,FN,TP - tuple : 4 list of TN, FP, FN, TP. In each of them, one element for one threshold
    '''
    from sklearn.metrics import confusion_matrix
    
    TN = [confusion_matrix(y_true=y_true,y_pred=y_pred_proba>=th).ravel()[0] for th in thresholds]
    FP = [confusion_matrix(y_true=y_true,y_pred=y_pred_proba>=th).ravel()[1] for th in thresholds]
    FN = [confusion_matrix(y_true=y_true,y_pred=y_pred_proba>=th).ravel()[2] for th in thresholds]
    TP = [confusion_matrix(y_true=y_true,y_pred=y_pred_proba>=th).ravel()[3] for th in thresholds]
    
    return TN,FP,FN,TP






def meanCostPerPrediction (confusionMatrix,costMatrix) :
    
    '''
    average cost per prediction, given aconfusionMatrix, and a cost matrix
    
    parameters :
    ------------
    confusionMatrix - array : scikit learn style BINARY confusion matrix :
                                TN  FP
                                FN  TP  
    costMatrix - array : BINARY classification cost matrix :
                            CTN  CFP
                            CFN  CTP
    
    returns :
    ---------
    averageCostPerPrediction - float : average cost per predicition
    
    '''
    
    import numpy as np
    
    # extract TN,FP,FN,TP
    TN,FP,FN,TP=confusionMatrix.ravel()
    Total=TN+FP+FN+TP # compute number of predictions
    
    # extract  cTN,cFP,cFN,cTP
    if type(costMatrix) != np.ndarray :
        costMatrix=np.array(costMatrix)
    cTN,cFP,cFN,cTP=costMatrix.ravel()
    
    # compute rates
    TNR=TN/(TN+FP)
    FPR=FP/(TN+FP)
    FNR=FN/(TP+FN)
    TPR=TP/(TP+FN)
    
    # compute percentage of positives and negatives
    actualPositiveRatio = (TP+FN)/Total
    actualNegativeRatio = (TN+FP)/Total
    
    # compute average cost per prediction
    averageCostPerPrediction= cTN * TNR * actualNegativeRatio \
                            + cFP * FPR * actualNegativeRatio \
                            + cFN * FNR * actualPositiveRatio \
                            + cTP * TPR * actualPositiveRatio
    
    return averageCostPerPrediction






def classificationCostsDf(y_true,y_proba,costMatrix,drop_intermediate=False) :
    
    '''
    return the different average costs per prediciton, for given thresholds extracted from the scikit learn roc_curve function
    
    parameters :
    ------------
    y_pred_proba - list : prediction scores
    y_true - list : true values 
    costMatrix - array : BINARY classification cost matrix :
                            CTN  CFP
                            CFN  CTP
    
    drop_intermediate - bool : roc_curve function parameter. BY DEFAULT : False
    
    return :
    --------
    df - dataframe : one column with diffrents thresholds extracted from roc_curve, one column with costs
    
    '''
    
    import pandas as pd
    from sklearn.metrics import roc_curve
    from sklearn.metrics import confusion_matrix
    import gc
    
    # extract thresholds
    FPR,RECALL,TH = roc_curve(y_score=y_proba,y_true=y_true,drop_intermediate=drop_intermediate)
    if TH[0]==float("inf") : # handle float infinity
        newMaxTh = round(TH[1],1) if round(TH[1],1)>TH[1] else round(TH[1],1)+0.1
        TH = [newMaxTh]+list(TH)[1:] 
    
    # compute costs using meanCostPerPrediction() function, for each threshold
    costs = [ meanCostPerPrediction ( 
                                        confusion_matrix(
                                                            y_true=y_true,
                                                            y_pred=y_proba >= threshold
                                                        ),
                                        costMatrix
                                    )
             for threshold in TH
            ]
    
    # put results in the dataframe
    df = pd.DataFrame({"Threshold":TH,"Average_prediction_cost":costs})
    
    del FPR,RECALL,TH,costs
    gc.collect()
    
    return df








def getCostEffectiveClassifModelUsingTEST(alreadyFittedModelsList,
                                             namesList,
                                             X_test,
                                             y_test,
                                             costMatrix,
                                             drop_intermediate=False,
                                             plot=True,
                                             palette=None) :
    
    '''
    for binary classification
    from different ALREADY fitted models, give the best cost effective one for a given cost matrix, with the optimum threshold
    
    parameters :
    ------------
    alreadyFittedModelsList - list or model : list of ALREADY fitted models (or only one model)
    namesList - list or string : list of names (or only one name)
    X_test - array or dataframe : testing data
    y_test - array or Series : target values for testing data
    costMatrix - array : BINARY classification cost matrix :
                            CTN  CFP
                            CFN  CTP
    drop_intermediate - bool : roc_curve function parameter. BY DEFAULT : False
    plot - bool : whether or not plot costs/thresholds curve. BY DEFAULT : True
    palette - list : list of colors. Default : None, use of seaborn "tab10" palette
    
    return :
    --------
    lowestMinCostModel - model : the best fitted model
    lowestMinCostThreshold - float : the optimal threshold
    lowestMinCost - float : the minimum average cost per prediction (for this threshold)
    '''
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    
    # create figure and handle palette
    if plot == True :
        fig,axs = plt.subplots(1,1,figsize=(12,4))
        if not palette :
            palette = sns.color_palette("tab10")
    
    # handle alreadyFittedModelsList and namesList types
    if type(alreadyFittedModelsList)!=list :
        alreadyFittedModelsList=[alreadyFittedModelsList]
    if type(namesList)!=list :
        namesList=[namesList]
    
    # initiate a dataframe to store costs
    minCostDf=pd.DataFrame()
    
    # iterate on models
    for i,clf in enumerate(alreadyFittedModelsList) :
        
        # predict scores on test set
        if hasattr(clf, "predict_proba"):
            y_pred_proba = clf.predict_proba(X_test)[:, 1]
        elif hasattr(clf, "decision_function"):
            y_pred_proba = clf.decision_function(X_test)
        else:
            y_pred_proba = clf.predict(X_test) 
        
        # using predictions scores, compute the cost table 
        costsDfClf=classificationCostsDf(y_true=y_test,
                                         y_proba=y_pred_proba,
                                         costMatrix=costMatrix,
                                         drop_intermediate=drop_intermediate)
        
        # get minimun cost for this clf, and threshold
        minCostThClf = costsDfClf["Threshold"][costsDfClf["Average_prediction_cost"].idxmin()]
        minCostClf = costsDfClf["Average_prediction_cost"].min()
        
        # put results in minCostDf
        minCostDf.loc[i,"model"]=namesList[i]
        minCostDf.loc[i,"threshold"]=minCostThClf
        minCostDf.loc[i,"cost"]=minCostClf
        
        # add this clf to a plot cost/threshold
        if plot == True :
            sns.lineplot(data=costsDfClf,x="Threshold",y="Average_prediction_cost",ax=axs,label=namesList[i],color=palette[i]);
            sns.scatterplot(x=[minCostThClf],y=[minCostClf],color=palette[i],marker=6)

    # set title and diplay
    if plot == True :
        axs.set_title("Average prediction cost per threshold")
        plt.show() 
    display(minCostDf)
    
    # compare each clf on select best one
    idxmin = minCostDf["cost"].idxmin()
    lowestMinCost = minCostDf["cost"].min()
    lowestMinCostModel = alreadyFittedModelsList[idxmin]
    lowestMinCostThreshold = minCostDf["threshold"][idxmin]
    print("le modèle avec le coût le plus faible est ",minCostDf["model"][idxmin])
    
    return lowestMinCostModel,lowestMinCostThreshold,lowestMinCost


        
    







def getCostEffectiveClassifModelUsingCV(notFittedModelsList,
                                 namesList,
                                 XTrain,
                                 yTrain,
                                 costMatrix,
                                 kf=None,
                                 drop_intermediate=False,
                                 plot=True,
                                 palette=None) :
    
    '''
    for binary classification
    from different models, NOT FITTED, give the best cost effective one for a given cost matrix, with the optimum threshold, 
    using Cross Validation
    
    parameters :
    ------------
    notFittedModelsList - list or model : list of NOT fitted models (or only one model)
    namesList - list or string : list of names (or only one name)
    XTrain - array or dataframe : training data
    yTrain - array or Series : target values for training data
    costMatrix - array : BINARY classification cost matrix :
                            CTN  CFP
                            CFN  CTP
    kf - cross-validator or int : for computing cross validation ypreds
                                    cross-validator : use kf
                                    int : use stratified kfold with n_splits=kf, shuffle=True, random_state=1
                                    BY DEFAULT : None, use stratified kfold with n_splits=5, shuffle=True, random_state=1
    drop_intermediate - bool : roc_curve function parameter. BY DEFAULT : False
    plot - bool : whether or not plot costs/thresholds curve. BY DEFAULT : True
    palette - list : list of colors. BY DEFAULT : None, use of seaborn "tab10" palette
    
    return :
    --------
    lowestMinCostModel - model : the best fitted model
    lowestMinCostThreshold - float : the optimal threshold
    lowestMinCost - float : the minimum average cost per prediction (for this threshold)
    '''
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    
    # create figure and handle palette
    if plot == True :
        fig,axs = plt.subplots(1,1,figsize=(12,4))
        if not palette :
            palette = sns.color_palette("tab10")
    
    # handle notFittedModelsList and namesList types
    if type(notFittedModelsList)!=list :
        notFittedModelsList=[notFittedModelsList]
    if type(namesList)!=list :
        namesList=[namesList]
    
    # handle kf
    if not kf :
        kf=StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    if type(kf)==int :
        kf=StratifiedKFold(n_splits=kf, shuffle=True, random_state=1)
    
    # initiate a dataframe to store CV preds
    CVpreds=pd.DataFrame(columns=namesList,index=range(len(XTrain)))
    # initiate a dataframe to store costs
    minCostDf=pd.DataFrame()
    
    # iterate on models
    for i,clf in enumerate(notFittedModelsList) :
        # iterate on folds
        for n_fold, (train_idx,valid_idx) in enumerate(kf.split(XTrain,yTrain)) :
            
            X_trainCV, y_trainCV = XTrain[train_idx],yTrain[train_idx]
            X_validCV, y_validCV = XTrain[valid_idx],yTrain[valid_idx]
            
            # fit the model on train CV set
            clf.fit(X_trainCV,y_trainCV)
            
            # predict scores on valid CV set and put them in the df
            if hasattr(clf, "predict_proba"):
                CVpreds.loc[valid_idx,namesList[i]] = clf.predict_proba(X_validCV)[:, 1]
            elif hasattr(clf, "decision_function"):
                CVpreds.loc[valid_idx,namesList[i]] = clf.decision_function(X_validCV)
            else:
                CVpreds.loc[valid_idx,namesList[i]] = clf.predict(X_validCV) 

        # using predictions scores, compute the cost table        
        costsDfClf=classificationCostsDf(y_true=yTrain,
                                         y_proba=CVpreds[namesList[i]],
                                         costMatrix=costMatrix,
                                         drop_intermediate=drop_intermediate)
        
        # get minimun cost for this clf, and threshold
        minCostThClf = costsDfClf["Threshold"][costsDfClf["Average_prediction_cost"].idxmin()]
        minCostClf = costsDfClf["Average_prediction_cost"].min()
        
        # put results in minCostDf
        minCostDf.loc[i,"model"]=namesList[i]
        minCostDf.loc[i,"threshold"]=minCostThClf
        minCostDf.loc[i,"cost"]=minCostClf
        
        # add this clf to a plot cost/threshold
        if plot == True :
            sns.lineplot(data=costsDfClf,x="Threshold",y="Average_prediction_cost",ax=axs,label=namesList[i],color=palette[i]);
            sns.scatterplot(x=[minCostThClf],y=[minCostClf],color=palette[i],marker=6)
    # set title and diplay
    if plot == True :
        axs.set_title("Average prediction cost per threshold")
        plt.show() 
    display(minCostDf)
    
    # compare each clf on select best one
    idxmin = minCostDf["cost"].idxmin()
    lowestMinCost = minCostDf["cost"].min()
    lowestMinCostModelFITTED = notFittedModelsList[idxmin].fit(XTrain,yTrain)
    lowestMinCostThreshold = minCostDf["threshold"][idxmin]
    print("le modèle avec le coût le plus faible est ",minCostDf["model"][idxmin])
    
    return lowestMinCostModelFITTED,lowestMinCostThreshold,lowestMinCost




def plotROCandPR(alreadyFittedModelsList,namesList,Xtest,ytest,plot_chance_level=False,palette=None) :
    '''
    plot ROC curve and PrecisionRecall curve for given model(s), using predictions on a test cv
    
    parameters :
    ------------
    alreadyFittedModelsList - list of models, or model type : a list of ALREADY fitted models (or one ALREADY model)
    namesList - list of string, or string : a list of names, one for each model (or one model name)
    Xtest - array or dataframe : testing data
    ytest - array or Series : target values for testing data
    plot_chance_level - bool : whether or not to plot random classifier curve. BY DEFAULT : False
    palette - list : list of colors. BY DEFAULT : None (use of seaborn "tab10")
    
    output :
    --------
    display curves
    
    '''
    
    # imports
    from sklearn.metrics import RocCurveDisplay
    from sklearn.metrics import PrecisionRecallDisplay
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    
    # create figure, both axes and palette
    fig,axs = plt.subplots(1,2,figsize=(12,6))
    fig.suptitle("ROC and Precision-Recall curves\nfrom preds on test set")
    axs=axs.flatten()
    if not palette :
        palette = sns.color_palette("tab10")
    
    # handle inputs if type is not list
    if type(alreadyFittedModelsList)!=list :
        alreadyFittedModelsList=[alreadyFittedModelsList]
    if type(namesList)==str :
        namesList=[namesList]
    
    # add ROC and PR on respectives axes
    for i,model in enumerate(alreadyFittedModelsList) :
        RocCurveDisplay.from_estimator(estimator=model,
                                       X=Xtest,
                                       y=ytest,
                                       ax=axs[0],
                                       plot_chance_level=plot_chance_level if i==0 else False,
                                       name=namesList[i],
                                       color=palette[i],
                                       alpha=0.8
                                      )
        PrecisionRecallDisplay.from_estimator(estimator=model,
                                              X=Xtest,
                                              y=ytest,
                                              ax=axs[1],
                                              plot_chance_level=plot_chance_level if i==0 else False,
                                              name=namesList[i],
                                              color=palette[i],
                                              alpha=0.8
                                             )

    # setting ylim and legend    
    for ax in axs :
        ax.axis("square")
        ax.set_ylim(-0.05,1.05)
        ax.set_xlim(-0.05,1.05)

        
    axs[0].legend(loc="lower right")
    axs[1].legend(loc="best")

        
    plt.show()







def plotROCandPRfromCV (oofProb, modelName, Xtrain, ytrain, kf, style="mean", plot_chance_level=False,palette=None) :
    '''
    
    plot ROC curve and PrecisionRecall curve for 1 given model, using out of folds scores from a cross validation
    
    parameters :
    ------------
    oofProb - probs obtained from CV
    modelName - string 
    Xtrain - array or dataframe : training data
    yTrain - array or Series : target values for training data, the ones used to obtain cvProbs
    kf - cross-validator : the one used to obtain cvProbs
    style - string : "mean" or "oof"
                        "mean" - plot global curve using the mean of each fold curve
                        "oof" - plot global curve using full oofProb and full ytrain to plot a new curve
    plot_chance_level - bool
    palette - list : list of colors. Default : None, use of seaborn "tab10" palette
    
    output :
    --------
    display curves
    
    '''

    # imports
    from sklearn.metrics import RocCurveDisplay, auc
    from sklearn.metrics import PrecisionRecallDisplay
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    
    # create figure, both axes and palette
    fig,axs = plt.subplots(1,2,figsize=(12,7))
    axs=axs.flatten()
    if style=="mean" :
        fig.suptitle(modelName+"\nROC and Precision-Recall curves\nfrom Cross Validation\n(using the mean of folds curves)")
    if style=="oof" :
        fig.suptitle(modelName+"\nROC and Precision-Recall curves\nfrom Cross Validation\n(using full out-of-fold preds)")
    if not palette :
        palette = sns.color_palette("tab10")
    
    # create x axis values
    linFPR = np.linspace(0, 1, 1000)
    linRecall = np.linspace(0, 1, 1000)
    
    # create lists to receive...
    TPRs = [] #... folds recalls for ROC curve
    aucs=[] #... folds roc aucs
    precisions = [] # ... folds precisions for PR curve
    a_ps=[] # ... folds average precisions




    # iterate on folds
    for k_fold, (train_idx,valid_idx) in enumerate(kf.split(Xtrain,ytrain)) :

        cvProb = oofProb[valid_idx]
        yTrue = ytrain[valid_idx]


        # for each fold, plot roc curve
        vizRoc = RocCurveDisplay.from_predictions(
                                   y_true=yTrue, 
                                   y_pred=cvProb,
                                   ax=axs[0],
                                   plot_chance_level=plot_chance_level if k_fold==0 else False,
                                   name="fold "+str(k_fold),
                                   color=palette[k_fold],
                                   lw=0.3
                                  )
        # store TPR values interpolated to match our xavis values
        TPRs_interpolations = np.interp(linFPR, vizRoc.fpr, vizRoc.tpr)
        TPRs_interpolations[0]=0
        TPRs.append(TPRs_interpolations)
        # store auc
        aucs.append(vizRoc.roc_auc)


        # for each fold, plot PR curve
        vizPR = PrecisionRecallDisplay.from_predictions(
                                   y_true=yTrue, 
                                   y_pred=cvProb,
                                   ax=axs[1],
                                   plot_chance_level=plot_chance_level if k_fold==0 else False,
                                   name="fold "+str(k_fold),
                                   color=palette[k_fold],
                                   lw=0.3
                                  )
        # store precisions values interpolated to match our xavis values
        recall=vizPR.recall
        recall=np.append(recall,0)
        precision=vizPR.precision
        precision=np.append(ytrain.mean(),precision)
        precisions_interpolations = np.interp(linRecall, np.flip(recall), np.flip(precision))
        precisions_interpolations[0]=1
        precisions.append(precisions_interpolations)
        # store average precision
        a_ps.append(vizPR.average_precision)

    # for ROC curve, use the mean of stored TPRs
    if style == "mean" :
        meanTPR = np.mean(TPRs, axis=0)
        meanTPR[-1] = 1

        mean_auc = auc(linFPR, meanTPR)
        std_auc = np.std(aucs)
        # plot mean ROC curve
        axs[0].plot(
            linFPR,
            meanTPR,
            color="b",
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=1.5,
            alpha=0.8
        )
        # plot std range
        std_TPR = np.std(TPRs, axis=0)
        TPRs_upper = np.minimum(meanTPR + std_TPR, 1)
        TPRs_lower = np.maximum(meanTPR - std_TPR, 0)
        axs[0].fill_between(
            linFPR,
            TPRs_lower,
            TPRs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
         )


    # for ROC curve, use the full out of folds scores to plot
    if style == "oof" :
        vizRocOof=RocCurveDisplay.from_predictions(
                                   y_true=ytrain, 
                                   y_pred=oofProb,
                                   ax=axs[0],
                                   plot_chance_level=False,
                                   name="full oof",
                                   color="green",
                                   lw=1.5,
                                   alpha=0.8
                                  )
        # plot std range using same method
        oofTPRs_interpolations = np.interp(linFPR, vizRocOof.fpr, vizRocOof.tpr)
        oofTPRs_interpolations[0]=0
        std_TPR = np.std(TPRs, axis=0)
        TPRs_upper = np.minimum(oofTPRs_interpolations + std_TPR, 1)
        TPRs_lower = np.maximum(oofTPRs_interpolations - std_TPR, 0)
        axs[0].fill_between(
            linFPR,
            TPRs_lower,
            TPRs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
         )


    # for PR curve, use the mean of stored precisions
    if style == "mean" :
        meanPrecision = np.mean(precisions,axis=0)
        meanPrecision[-1] = ytrain.mean()

        mean_a_p = np.sum(np.diff(linRecall) * np.array(meanPrecision)[:-1])
        std_a_p = np.std(mean_a_p)
        # plot mean PR curve
        axs[1].plot(
            linRecall,
            meanPrecision,
            color="b",
            label=r"Mean PR (AP = %0.2f $\pm$ %0.2f)" % (mean_a_p, std_a_p),
            lw=1.5,
            alpha=0.8,
        )
        # plot std range
        std_precision = np.std(precisions, axis=0)
        precisions_upper = np.minimum(meanPrecision + std_precision, 1)
        precisions_lower = np.maximum(meanPrecision - std_precision, 0)
        axs[1].fill_between(
            linRecall,
            precisions_lower,
            precisions_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
         )
    # for PR curve, use the full out of folds scores to plot
    if style == "oof" :
        vizPROof=PrecisionRecallDisplay.from_predictions(
                                   y_true=ytrain, 
                                   y_pred=oofProb,
                                   ax=axs[1],
                                   plot_chance_level=False,
                                   name="full oof",
                                   color="green",
                                   lw=1.5,
                                   alpha = 0.8
                                  )
        # plot std range using same method
        oofPrecisions_interpolations = np.interp(linRecall, np.flip(vizPROof.recall), np.flip(vizPROof.precision))
        oofPrecisions_interpolations[0]=1
        std_precision = np.std(precisions, axis=0)
        precisions_upper = np.minimum(oofPrecisions_interpolations + std_precision, 1)
        precisions_lower = np.maximum(oofPrecisions_interpolations - std_precision, 0)
        axs[1].fill_between(
            linRecall,
            precisions_lower,
            precisions_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
         )

    
    # set axis form and limits
    for ax in axs :
        ax.axis("square")
        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            )



    # set titles and legends
    axs[0].legend(loc="lower right")
    axs[1].legend(loc="best")
    if style == "mean" :
        axs[0].set_title("Mean ROC curve with variability")
        axs[1].set_title("Mean PR curve with variability")
    if style == "oof" :
        axs[0].set_title("full oof ROC curve with variability")
        axs[1].set_title("full oof PR curve with variability")

    plt.show()
    
    







def plotROCandPRfromCV_several (oofProbList, modelNameList, Xtrain, ytrain, kf, style="mean", plot_chance_level=False,palette=None) :
    '''
    plot ROC curve and PrecisionRecall curve for given models, using out of folds scores from a cross validation

    oofProbList - list : list of probs obtained from CV
    modelNameList - list : list of string 
    Xtrain - array or dataframe : training data
    yTrain - array or Series : target values for training data, the ones used to obtain cvProbs
    kf - cross-validator : the one used to obtain cvProbs
    style - string : "mean" or "oof"
                        "mean" - plot global curves using the mean of each fold curve
                        "oof" - plot global curves using each full oofProb and full ytrain to plot the curves
    plot_chance_level - bool
    palette - list : list of colors. Default : None, use of seaborn "tab10" palette
    
    
    
    '''

    # imports
    from sklearn.metrics import RocCurveDisplay, auc,roc_curve
    from sklearn.metrics import PrecisionRecallDisplay,precision_recall_curve
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    
    # create figure, both axes and palette
    fig,axs = plt.subplots(1,2,figsize=(12,7))
    axs=axs.flatten()
    if style=="mean" :
        fig.suptitle("ROC and Precision-Recall curves\nfrom Cross Validation\n(using the mean of folds curves)")
    if style=="oof" :
        fig.suptitle("ROC and Precision-Recall curves\nfrom Cross Validation\n(using full out-of-fold preds)")

    if not palette :
        palette = sns.color_palette("tab10")


    # iterate on models
    for i, (oofProb,modelName) in enumerate(zip(oofProbList,modelNameList)) :
        
        # if use the mean of folds curves, we iterate on folds
        if style == "mean" :
            # create x axis values and list to store folds y values
            # for roc
            linFPR = np.linspace(0, 1, 1000)
            TPRs = []
            # for PR
            linRecall = np.linspace(0, 1, 1000)
            precisions = []

            # iterate on folds
            for k_fold, (train_idx,valid_idx) in enumerate(kf.split(Xtrain,ytrain)) :

                cvProb = oofProb[valid_idx]
                yTrue = ytrain[valid_idx]


                # extract FPR and TPR of each fold roc curve
                FPR,TPR,_ = roc_curve(
                                           y_true=yTrue, 
                                           y_score=cvProb,
                                           drop_intermediate=False,
                                          )
                # store TPRs values interpolated to match our xavis values
                TPRs_interpolations = np.interp(linFPR, FPR, TPR)
                TPRs_interpolations[0]=0
                TPRs.append(TPRs_interpolations)



                # extract precision and recall of each fold PR curve
                precision, recall, _ = precision_recall_curve(
                                           y_true=yTrue, 
                                           probas_pred=cvProb,
                                           drop_intermediate=False,
                                          )
                
                # store precisions values interpolated to match our xavis values
                recall=np.append(recall,0)
                precision=np.append(ytrain.mean(),precision)

                precisions_interpolations = np.interp(linRecall, np.flip(recall), np.flip(precision))
                precisions_interpolations[0]=1

                precisions.append(precisions_interpolations)


            # for roc curve, use the mean of stored TPRs
            meanTPR = np.mean(TPRs, axis=0)
            meanTPR[-1] = 1
            mean_auc = auc(linFPR, meanTPR)

            axs[0].plot(
                linFPR,
                meanTPR,
                color=palette[i],
                label=modelName+" (mean AUC ="+str(round(mean_auc,2))+")",
                lw=1,
                alpha=0.8
            )
            # for PR curve, use the mean of stored precisions
            meanPrecision = np.mean(precisions,axis=0)
            meanPrecision[-1] = ytrain.mean()
            mean_a_p = np.sum(np.diff(linRecall) * np.array(meanPrecision)[:-1])

            axs[1].plot(
                linRecall,
                meanPrecision,
                color=palette[i],
                label=modelName+" (mean AP ="+str(round(mean_a_p,2))+")",
                lw=1,
                alpha=0.8,
            )
        # if we use the full "out of folds" scores :
        if style == "oof" :
            RocCurveDisplay.from_predictions(
                                       y_true=ytrain, 
                                       y_pred=oofProb,
                                       ax=axs[0],
                                       plot_chance_level=plot_chance_level,
                                       name=modelName,
                                       color=palette[i],
                                       lw=1,
                                       alpha=0.8
                                      )


            x=PrecisionRecallDisplay.from_predictions(
                                       y_true=ytrain, 
                                       y_pred=oofProb,
                                       ax=axs[1],
                                       plot_chance_level=plot_chance_level,
                                       name=modelName,
                                       color=palette[i],
                                       lw=1,
                                       alpha = 0.8
                                      )


    # set axis form and limits
    for ax in axs :
        ax.axis("square")
        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            )



    # set titles and legends
    axs[0].legend(loc="lower right")
    axs[1].legend(loc="best")
    if style == "mean" :
        axs[0].set_title("mean ROC curve")
        axs[1].set_title("mean PR curve")
        axs[0].set_ylabel("True Positive Rate (Positive label: 1)")
        axs[0].set_xlabel("False Positive Rate (Positive label: 1)")
        axs[1].set_ylabel("Precision (Positive label: 1)")
        axs[1].set_xlabel("Recall (Positive label: 1)")
    if style == "oof" :
        axs[0].set_title("full oof ROC curve")
        axs[1].set_title("full oof PR curve")


    plt.show()
    



def plotROCandPRfromCV_WID (oofProbList, modelNameList, Xtrain, ytrain, kf,palette=None) :
    '''
    plot ROC curve and PrecisionRecall curve for 1 or several models, using out of folds scores from a cross validation

    use custom plotROCandPRfromCV_several() and plotROCandPRfromCV() functions

    oofProbList - list : list of probs obtained from CV
    modelNameList - list : list of string 
    Xtrain - array or dataframe : training data
    yTrain - array or Series : target values for training data, the ones used to obtain cvProbs
    kf - cross-validator : the one used to obtain cvProbs
    palette - list : list of colors. Default : None, use of seaborn "tab10" palette
    
    
    
    '''

    # imports
    import ipywidgets as wid

    
    # widget to select model 
    widModel=wid.Dropdown(options=["all"]+[modelName for modelName in modelNameList],
                              value="all",
                              description="Model :",
                              disabled=False,
                              style={'description_width': 'initial'}
                              )
    
    
    def plotSingleWid(oofProb, modelName, Xtrain, ytrain, kf,palette) :
        # imports
        import ipywidgets as wid
        # widget for style
        widStyle=wid.RadioButtons(options={"use the full 'out of folds' pred":"oof","use the mean of folds curves":"mean"},
                              value="mean",
                              description="Method :",
                              disabled=False,
                              style={'description_width': 'initial'}
                              )
        
        # widget for chance level
        widChance=wid.RadioButtons(options={"yes":True,"no":False},
                                  value=False,
                                  description="Plot chance level :",
                                  disabled=False,
                                  style={'description_width': 'initial'}
                                  )
        
        ui=wid.HBox([widStyle,widChance])
        
        # oofProb, modelName, Xtrain, ytrain, kf, style="mean", plot_chance_level=False,palette=None
        out=wid.interactive_output(plotROCandPRfromCV,
                                   {
                                       "oofProb" : wid.fixed(oofProb),
                                       "modelName" : wid.fixed(modelName),
                                       "Xtrain" :wid.fixed(Xtrain),
                                       "ytrain" :wid.fixed(ytrain),
                                       "kf" :wid.fixed(kf),
                                       "style" : widStyle,
                                       "plot_chance_level" : widChance,
                                       "palette" : wid.fixed(palette)
                                   })
        
        display(ui,out)
    
    def plotMultiWid(oofProbList,modelNameList,Xtrain,ytrain,kf,palette) :
        # imports
        import ipywidgets as wid
        # widget for style
        widStyle=wid.RadioButtons(options={"use the full 'out of folds' pred":"oof","use the mean of folds curves":"mean"},
                              value="mean",
                              description="Method :",
                              disabled=False,
                              style={'description_width': 'initial'}
                              )
        
        # widget for chance level
        widChance=wid.RadioButtons(options={"no":False,"yes":True},
                                  value=False,
                                  description="Plot chance level :",
                                  disabled=False,
                                  style={'description_width': 'initial'}
                                  )
        
        # handle widStyle change
        def handleStyleChange(change) :
            if change.new=="oof" :
                widChance.options={"no":False,"yes":True}
            if change.new=="mean" :
                widChance.options={"no":False}
        widStyle.observe(handleStyleChange,'value')
        
        ui=wid.HBox([widStyle,widChance])
        
        out = wid.interactive_output(plotROCandPRfromCV_several,
                                   {
                                       "oofProbList" : wid.fixed(oofProbList),
                                       "modelNameList" : wid.fixed(modelNameList),
                                       "Xtrain" :wid.fixed(Xtrain),
                                       "ytrain" :wid.fixed(ytrain),
                                       "kf" :wid.fixed(kf),
                                       "style" : widStyle,
                                       "plot_chance_level" : widChance,
                                       "palette" : wid.fixed(palette)
                                   })
        display(ui,out)
        
    def whichSingleOrMulti(model) :
        if model == "all" :
            plotMultiWid(oofProbList=oofProbList,modelNameList=modelNameList,
                         Xtrain=Xtrain,ytrain=ytrain,kf=kf,palette=palette)
        else :
            plotSingleWid(oofProb=oofProbList[modelNameList.index(model)], modelName=model, 
                         Xtrain=Xtrain,ytrain=ytrain,kf=kf,palette=palette)
            
    out = wid.interactive_output(whichSingleOrMulti,{"model":widModel})
    
    display(widModel,out)