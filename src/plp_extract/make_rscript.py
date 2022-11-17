def renew_run_script(train_conf, client_conf):
    cfg = client_conf
    run_script = (
        f"""
    library({train_conf.package})
    library(PatientLevelPrediction)
    library(dplyr)
    setwd("./temps")
    # USER INPUTS
    #=======================
    # The folder where the study intermediate and result files will be written:
    outputFolder <- "./{train_conf.project}Results"

    # Specify where the temporary files (used by the ff package) will be created:
    options(fftempdir = "{cfg.fftempdir}")

    # Details for connecting to the server:
    dbms <- "{cfg.dbms}"
    user <- '{cfg.user}'
    pw <- '{cfg.pw}'
    server <- '{cfg.server}'
    port <- '{cfg.port}'
    connectionDetails <- DatabaseConnector::createConnectionDetails(dbms = dbms,
                                                                    server = server,
                                                                    user = user,
                                                                    password = pw,
                                                                    port = port,
                                                                    pathToDriver = '{cfg.pathToDriver}')

    # Add the database containing the OMOP CDM data
    cdmDatabaseSchema <- '{cfg.cdmDatabaseSchema}'
    # Add a sharebale name for the database containing the OMOP CDM data
    cdmDatabaseName <- '{cfg.cdmDatabaseName}'
    # Add a database with read/write access as this is where the cohorts will be generated
    cohortDatabaseSchema <- '{cfg.cohortDatabaseSchema}'

    oracleTempSchema <-NULL # {cfg.oracleTempSchema}, if oracle is used

    # table name where the cohorts will be generated
    cohortTable <- "{cfg.cohortTable}"

    #=======================

    execute(connectionDetails = connectionDetails,
            cdmDatabaseSchema = cdmDatabaseSchema,
            cdmDatabaseName = cdmDatabaseName,
            cohortDatabaseSchema = cohortDatabaseSchema,
            oracleTempSchema = oracleTempSchema,
            cohortTable = cohortTable,
            outputFolder = outputFolder,
            createProtocol = F,
            createCohorts = T,
            runAnalyses = F,
            createResultsDoc = F,
            packageResults = F,
            createValidationPackage = F,  
            #analysesToValidate = 1,
            minCellCount= 5,
            createShiny = F,
            createJournalDocument = F,
            analysisIdDocument = 1)

    #=======================
    verbosity = "INFO"
    cdmVersion = 5

    ParallelLogger::logInfo("Running predictions")
    predictionAnalysisListFile <- system.file("settings",
                                            "predictionAnalysisList.json",
                                            package = "{train_conf.package}")
    predictionAnalysisList <- PatientLevelPrediction::loadPredictionAnalysisList(predictionAnalysisListFile)
    predictionAnalysisList$connectionDetails = connectionDetails
    predictionAnalysisList$cdmDatabaseSchema = cdmDatabaseSchema
    predictionAnalysisList$cdmDatabaseName = cdmDatabaseName
    predictionAnalysisList$oracleTempSchema = oracleTempSchema
    predictionAnalysisList$cohortDatabaseSchema = cohortDatabaseSchema
    predictionAnalysisList$cohortTable = cohortTable
    predictionAnalysisList$outcomeDatabaseSchema = cohortDatabaseSchema
    predictionAnalysisList$outcomeTable = cohortTable
    predictionAnalysisList$cdmVersion = cdmVersion
    predictionAnalysisList$outputFolder = outputFolder
    predictionAnalysisList$verbosity = verbosity

    # result <- do.call(PatientLevelPrediction::runPlpAnalyses, predictionAnalysisList)

    outcomeDatabaseSchema = predictionAnalysisList$outcomeDatabaseSchema
    outcomeTable = predictionAnalysisList$outcomeTable
    onlyFetchData = FALSE
    modelAnalysisList = predictionAnalysisList$modelAnalysisList
    cohortIds = predictionAnalysisList$cohortIds
    cohortNames = predictionAnalysisList$cohortNames
    outcomeIds = predictionAnalysisList$outcomeIds
    outcomeNames = predictionAnalysisList$outcomeNames
    washoutPeriod = predictionAnalysisList$washoutPeriod
    maxSampleSize = predictionAnalysisList$maxSampleSize
    minCovariateFraction = predictionAnalysisList$minCovariateFraction
    normalizeData = predictionAnalysisList$normalizeData
    testSplit = predictionAnalysisList$testSplit
    testFraction = predictionAnalysisList$testFraction
    splitSeed = predictionAnalysisList$splitSeed
    nfold = predictionAnalysisList$nfold
    settings = NULL


    # start log:
    #clearLoggerType("Multple PLP Log")
    """
        + """
    if(!dir.exists(outputFolder)){dir.create(outputFolder,recursive=T)}
    logFileName = paste0(outputFolder,'/plplog.txt')
    logger <- ParallelLogger::createLogger(name = "Multple PLP Log",
                                        threshold = verbosity,
                                        appenders = list(ParallelLogger::createFileAppender(layout = ParallelLogger::layoutParallel,
                                                                                            fileName = logFileName)))
    ParallelLogger::registerLogger(logger)

    if (missing(outcomeIds)){
    stop("Need to specify outcome ids")
    }
    if (missing(cohortIds)){
    stop("Need to specify cohort ids")
    }
    if (missing(connectionDetails)){
    stop("Need to specify connectionDetails")
    }
    if (missing(cdmDatabaseSchema)){
    stop("Need to specify cdmDatabaseSchema")
    }
    if (missing(cdmDatabaseName)){
    stop("Need to specify cdmDatabaseName - a shareable name for the database")
    }
    if (missing(modelAnalysisList)){
    stop("Need to specify modelAnalysisList")
    }
    # check input types
    plpDataSettings <- list(connectionDetails = connectionDetails,
                            cdmDatabaseSchema = cdmDatabaseSchema,
                            oracleTempSchema = oracleTempSchema, 
                            cohortDatabaseSchema = cohortDatabaseSchema,
                            cohortTable = cohortTable,
                            outcomeDatabaseSchema = outcomeDatabaseSchema,
                            outcomeTable = outcomeTable,
                            cdmVersion = cdmVersion,
                            firstExposureOnly = F,
                            washoutPeriod = washoutPeriod,
                            sampleSize = maxSampleSize
    )

    runPlpSettings <- list(minCovariateFraction = minCovariateFraction,
                        normalizeData = normalizeData,
                        testSplit = testSplit,
                        testFraction = testFraction,
                        splitSeed = splitSeed,
                        nfold = nfold,
                        verbosity = verbosity )

    if (!dir.exists(outputFolder)){
    dir.create(outputFolder)
    }
    if (!dir.exists(file.path(outputFolder,'Validation'))){
    dir.create(file.path(outputFolder,'Validation'), recursive = T)
    }



    createPlpReferenceTable <- function(modelAnalysisList,
                                        cohortIds,
                                        outcomeIds,
                                        outputFolder, cdmDatabaseName){
    
    #analysisId, cohortId, outcomeId, settingsFile, plpDataFolder, studyPopFile, plpResultFolder
    
    analyses <- expand.grid(cohortId = cohortIds,
                            outcomeId = outcomeIds,
                            modelSettingsId = modelAnalysisList$settingLookupTable$lookupId)
    # remove rows with same cohortId and outcomeId
    removeInd <- analyses$cohortId == analyses$outcomeId
    analyses <- analyses[!removeInd, ]
    
    analyses$analysisId <- 1:nrow(analyses)
    analyses$devDatabase <- cdmDatabaseName
    analyses <- merge(analyses, modelAnalysisList$settingLookupTable, 
                        by.x='modelSettingsId', by.y='lookupId', all.x=T)
    
    # TODO: replace outputFolder with '.' to make relative positions
    analyses$plpDataFolder <- file.path(outputFolder,
                                        paste0('PlpData_L',analyses$covariateSettingId,'_T',analyses$cohortId))
    analyses$studyPopFile <- file.path(outputFolder,
                                        paste0('StudyPop_L',analyses$covariateSettingId,'_T',analyses$cohortId,'_O',analyses$outcomeId,'_P',analyses$populationSettingId,'.rds'))
    analyses$plpResultFolder <- file.path(outputFolder,
                                            paste0('Analysis_',analyses$analysisId))
    return(analyses)  
    }
    ParallelLogger::logTrace(paste0('Creating reference table'))
    referenceTable <- tryCatch({createPlpReferenceTable(modelAnalysisList,
                                                        cohortIds,
                                                        outcomeIds,
                                                        outputFolder, cdmDatabaseName)},
                            error = function(cont){ParallelLogger::logTrace(paste0('Creating reference table error:', cont)); stop()})
    if(!missing(cohortNames)){
    if(!is.null(cohortNames))
        if(length(cohortNames)!=length(cohortIds)){
        stop('cohortNames entered but different length to cohortIds')
        }
    cnames <- data.frame(cohortId=cohortIds, cohortName=cohortNames)
    referenceTable <- merge(referenceTable, cnames, by='cohortId', all.x=T)
    }
    if(!missing(outcomeNames)){
    if(!is.null(outcomeNames))
        if(length(outcomeNames)!=length(outcomeIds)){
        stop('outcomeNames entered but different length to outcomeIds')
        }
    onames <- data.frame(outcomeId=outcomeIds, outcomeName=outcomeNames)
    referenceTable <- merge(referenceTable, onames, by='outcomeId', all.x=T)
    }

    # if settings are there restrict to these:
    if(!is.null(settings)){
    if(nrow(settings) != 0){
        ParallelLogger::logInfo('Restricting to specified settings...')
        
        # if transpose fix it 
        if(sum(row.names(settings)%in%c('cohortId', 'outcomeId', 'populationSettingId',
                                        'modelSettingId', 'covariateSettingId'))==5){
        settings <- t(settings)
        }
        
        referenceTable <- merge(settings, referenceTable, by = c('cohortId', 
                                                                'outcomeId', 'populationSettingId',
                                                                'modelSettingId', 'covariateSettingId'))
    }
    }

    if(!file.exists(file.path(outputFolder,'settings.csv'))){
    ParallelLogger::logTrace(paste0('Writing settings csv to ',file.path(outputFolder,'settings.csv') ))
    utils::write.csv(referenceTable,
                    file.path(outputFolder,'settings.csv'), 
                    row.names = F )
    }
    #########################
    i <- 1

    plpDataFolder <- referenceTable$plpDataFolder[i]
    if(!dir.exists(plpDataFolder)){
    ParallelLogger::logTrace(paste0('Running setting ', i ))
    
    oind <- referenceTable$cohortId==referenceTable$cohortId[i] & 
        referenceTable$covariateSettingId==referenceTable$covariateSettingId[i]
    outcomeIds <- unique(referenceTable$outcomeId[oind])
    
    plpDataSettings$cohortId <- referenceTable$cohortId[i]
    plpDataSettings$outcomeIds <- outcomeIds 
    plpDataSettings$covariateSettings <- modelAnalysisList$covariateSettings[[referenceTable$covariateSettingId[i]]]
    plpData <- tryCatch(do.call(PatientLevelPrediction::getPlpData, plpDataSettings),
                        finally= ParallelLogger::logTrace('Done plpData.'),
                        error= function(cond){ParallelLogger::logInfo(paste0('Error with getPlpData:',cond));return(NULL)})
    
    if(!is.null(plpData)){
        ParallelLogger::logTrace(paste0('Saving data in setting ', i ))
        savePlpData(plpData, referenceTable$plpDataFolder[i])
        #plpData <- loadPlpData(referenceTable$plpDataFolder[i])
    } else{
        ParallelLogger::logInfo('No plpData - probably empty cohort issue')
    }
    } else{
    ParallelLogger::logTrace(paste0('Loading data in setting ', i ))
    plpData <- PatientLevelPrediction::loadPlpData(referenceTable$plpDataFolder[i])
    }

    if(!file.exists(referenceTable$studyPopFile[i])){#studyPop[i])){
    ParallelLogger::logTrace(paste0('Setting population settings for setting ', i ))
    # get pop and save to referenceTable$popFile
    popSettings <- modelAnalysisList$populationSettings[[referenceTable$populationSettingId[i]]]
    popSettings$outcomeId <- referenceTable$outcomeId[i] 
    popSettings$plpData <- plpData
    population <- tryCatch(do.call(createStudyPopulation, popSettings),
                            finally= ParallelLogger::logTrace('Done pop.'), 
                            error= function(cond){ParallelLogger::logTrace(paste0('Error with pop:',cond));return(NULL)})
    if(!is.null(population)){
        ParallelLogger::logTrace(paste0('Saving population for setting ', i ))
        saveRDS(population, referenceTable$studyPopFile[i])#studyPop[i])
    }
    } else{
    ParallelLogger::logTrace(paste0('Loading population for setting', i ))
    population <- readRDS(referenceTable$studyPopFile[i])#studyPop[i])
    }

    plpResultFolder = file.path(referenceTable$plpResultFolder[i],'plpResult')
    if(!dir.exists(plpResultFolder) && !onlyFetchData){
    ParallelLogger::logTrace(paste0('Running runPlp for setting ', i ))
    dir.create(referenceTable$plpResultFolder[i], recursive = T)
    # runPlp and save result to referenceTable$plpResultFolder
    runPlpSettings$modelSettings <- modelAnalysisList$models[[referenceTable$modelSettingId[i]]]
    runPlpSettings$plpData <- plpData
    runPlpSettings$population <- population
    runPlpSettings$saveDirectory <- gsub(paste0('/Analysis_',referenceTable$analysisId[i]),'',referenceTable$plpResultFolder[i])
    runPlpSettings$analysisId <- paste0('Analysis_',referenceTable$analysisId[i])
    runPlpSettings$savePlpData <- F
    runPlpSettings$savePlpResult <- T
    runPlpSettings$savePlpPlots <- F
    runPlpSettings$saveEvaluation <- F
    
    }


    minCovariateFraction = runPlpSettings$minCovariateFraction
    normalizeData= runPlpSettings$normalizeData
    modelSettings = runPlpSettings$modelSettings
    testSplit = runPlpSettings$testSplit
    testFraction= runPlpSettings$testFraction
    trainFraction = NULL
    splitSeed=NULL
    nfold= runPlpSettings$nfold
    indexes= NULL
    saveDirectory=runPlpSettings$saveDirectory
    savePlpData=F
    savePlpResult=T
    savePlpPlots = F
    saveEvaluation = F
    timeStamp=FALSE
    analysisId=NULL
    runCovariateSummary = T
    save=NULL

    #################################
    ########  RunPlp.R code #########
    #################################


    if(!missing(save)){
    warning('save has been replaced with saveDirectory - please use this input from now on')
    if(is.null(saveDirectory)){saveDirectory <- save}
    }

    if(missing(verbosity)){
    verbosity <- "INFO"
    } else{
    if(!verbosity%in%c("DEBUG","TRACE","INFO","WARN","FATAL","ERROR", "NONE")){
        stop('Incorrect verbosity string')
    }
    }

    # log the start time:
    ExecutionDateTime <- Sys.time()

    # create an analysisid and folder to save the results
    start.all <- Sys.time()
    if(is.null(analysisId))
    analysisId <- gsub(':','',gsub('-','',gsub(' ','',start.all)))

    if(is.null(saveDirectory)){
    analysisPath <- file.path(getwd(),analysisId)
    } else {
    analysisPath <- file.path(saveDirectory,analysisId) 
    }

    if(verbosity!="NONE"){
    if(!dir.exists(analysisPath)){dir.create(analysisPath,recursive=T)}
    }
    logFileName = paste0(analysisPath,'/plplog.txt')


    # saveData <- 'Y'
    # if(tolower(saveData) %in% c('y', 'yes')){saveRDS(toSparseM(plpData, population), file = file.path(analysisPath, "CovariateMatrixData.rds"))}
    # if(tolower(saveData) %in% c('y', 'yes')){saveRDS(population, file = file.path(analysisPath, "population.rds"))}
    
    CovariateMatrixData <- toSparseM(plpData, population)
    df <- as.data.frame(as.matrix(CovariateMatrixData$data))
    colnames(df) <- CovariateMatrixData$covariateRef$covariateName
    df["rowId"] <- rownames(df)
    pop <- population[,c("rowId", "subjectId", "cohortStartDate", "cohortId", "ageYear", "gender", "outcomeCount", "timeAtRisk", "survivalTime")]
    pop_df <- merge(pop, df, by = "rowId")
    """
        + f"""
    arrow::write_parquet(pop_df, sink = file.path("cov_{train_conf.project}.parquet"))
    """
    )
    return run_script
