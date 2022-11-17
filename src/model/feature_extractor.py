"""
prediction folder 자체를 input으로 받음

0. 중간 일부분 코드 변경함
1. build 진행함
2. configuration 파일 받아서 codetorun.R 실행
3. patientLevelPrediction 중간에 feature extraction까지만 진행하기
4. 추출된 feature extractor를 download 진행함
"""

from loguru import logger
import os
import pandas as pd
import numpy as np
import shutil
from sklearn.preprocessing import MinMaxScaler
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
import time
from pathlib import Path
import decorator
from sklearn.model_selection import train_test_split

basics = [
    "rowId",
    "subjectId",
    "cohortStartDate",
    "cohortId",
    "ageYear",
    "gender",
    "outcomeCount",
    "timeAtRisk",
    "survivalTime",
]


def retry(howmany, *exception_types, **kwargs):
    timesleep = kwargs.get("timesleep", 10.0)  # seconds

    @decorator.decorator
    def tryIt(func, *fargs, **fkwargs):
        for _ in range(howmany):
            try:
                return func(*fargs, **fkwargs)
            except exception_types or Exception:
                if timesleep is not None:
                    time.sleep(timesleep)
                    logger.info(f"Retry {func.__name__}")

    return tryIt


def select_features_from_cov_pop(cov_pop, imp, common_cols, feature_num=1000):

    feature_sorted = (
        pd.DataFrame(imp.values()).fillna(0).mean().sort_values(ascending=False)
    )
    selected_feat = feature_sorted.index.tolist()
    selected_cols = cov_pop.columns[cov_pop.columns.isin(selected_feat)]
    selected_cols = selected_cols[selected_cols.isin(common_cols)].tolist()[
        :feature_num
    ]
    cov_pop = cov_pop[basics + selected_cols]
    return cov_pop


def format_to_ctr_external(
    external, sparse_features, dense_features
):  # TODO : scaler 정리하기
    # feature_selection 들어가있으면...
    fixlen_feature_columns = [
        SparseFeat(feat, 2)  # @note : 현재 categorical은 binary 변수밖에 못넣음
        for feat in sparse_features
    ] + [
        DenseFeat(
            feat,
            1,
        )
        for feat in dense_features
    ]
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    external_label = np.array(external["outcomeCount"])
    external_model_input = {name: external[name] for name in feature_names}

    key_with_all_plus = [
        key
        for key in external_model_input
        if (int(external_model_input[key].sum()) == len(external_model_input[key]))
    ]
    for key in key_with_all_plus:
        external_model_input[key] = pd.Series(np.zeros_like(external_model_input[key]))

    return (
        external_model_input,
        external_label,
        linear_feature_columns,
        dnn_feature_columns,
    )


def format_to_ctr(
    train, valid, test, sparse_features, dense_features
):  # TODO : scaler 정리하기
    # feature_selection 들어가있으면...
    fixlen_feature_columns = [
        SparseFeat(feat, 2)  # @note : 현재 categorical은 binary 변수밖에 못넣음
        for feat in sparse_features
    ] + [
        DenseFeat(
            feat,
            1,
        )
        for feat in dense_features
    ]
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    train_label = np.array(train["outcomeCount"])
    valid_label = np.array(valid["outcomeCount"])
    test_label = np.array(test["outcomeCount"])
    train_model_input = {name: train[name] for name in feature_names}
    valid_model_input = {name: valid[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    key_with_all_plus = [
        key
        for key in train_model_input
        if (int(train_model_input[key].sum()) == len(train_model_input[key]))
        | (int(valid_model_input[key].sum()) == len(valid_model_input[key]))
        | (int(test_model_input[key].sum()) == len(test_model_input[key]))
    ]
    for key in key_with_all_plus:
        train_model_input[key] = pd.Series(np.zeros_like(train_model_input[key]))
        valid_model_input[key] = pd.Series(np.zeros_like(valid_model_input[key]))
        test_model_input[key] = pd.Series(np.zeros_like(test_model_input[key]))

    return (
        train_model_input,
        valid_model_input,
        test_model_input,
        train_label,
        valid_label,
        test_label,
        linear_feature_columns,
        dnn_feature_columns,
    )


def split_patients(cov_pop, pct=None, random_state=328):
    if pct is None:
        pct = [0.7, 0.1, 0.2]

    train_pat, test_valid = train_test_split(
        cov_pop.subjectId.unique(), test_size=(1 - pct[0]), random_state=random_state
    )
    valid_pat, test_pat = train_test_split(
        test_valid, test_size=pct[2] / (pct[1] + pct[2]), random_state=random_state
    )
    train, valid, test = (
        cov_pop[cov_pop.subjectId.isin(train_pat)],
        cov_pop[cov_pop.subjectId.isin(valid_pat)],
        cov_pop[cov_pop.subjectId.isin(test_pat)],
    )

    return train, valid, test


def split_patients_outcome(cov_pop, pct=None, random_state=328):

    case_df = cov_pop.query("outcomeCount==1")
    cont_df = cov_pop.query("outcomeCount==0")

    case_df_pid = case_df.subjectId.unique()
    case_tr_pid, case_vd_pid = train_test_split(
        case_df_pid, test_size=0.3, random_state=random_state
    )
    case_vd_pid, case_te_pid = train_test_split(
        case_vd_pid, test_size=0.66, random_state=random_state
    )

    case_tr_df = case_df[case_df.subjectId.isin(case_tr_pid)]
    case_vd_df = case_df[case_df.subjectId.isin(case_vd_pid)]
    case_te_df = case_df[case_df.subjectId.isin(case_te_pid)]

    cont_in_case_tr = cont_df[cont_df.subjectId.isin(case_tr_pid)]
    cont_in_case_vd = cont_df[cont_df.subjectId.isin(case_vd_pid)]
    cont_in_case_te = cont_df[cont_df.subjectId.isin(case_te_pid)]

    cont_df_pid = cont_df[~cont_df.subjectId.isin(case_df.subjectId)].subjectId.unique()

    cont_tr_pid, cont_vd_pid = train_test_split(
        cont_df_pid, test_size=0.3, random_state=random_state
    )
    cont_vd_pid, cont_te_pid = train_test_split(
        cont_vd_pid, test_size=0.66, random_state=random_state
    )

    cont_tr_df = cont_df[cont_df.subjectId.isin(cont_tr_pid)]
    cont_vd_df = cont_df[cont_df.subjectId.isin(cont_vd_pid)]
    cont_te_df = cont_df[cont_df.subjectId.isin(cont_te_pid)]

    train = pd.concat([case_tr_df, cont_tr_df, cont_in_case_tr])
    valid = pd.concat([case_vd_df, cont_vd_df, cont_in_case_vd])
    test = pd.concat([case_te_df, cont_te_df, cont_in_case_te])
    return train, valid, test


def get_dense_sparse_features(cov_pop, except_cols=None):
    cov_pop = cov_pop.drop(except_cols, axis=1)
    unique_values = cov_pop.nunique()
    sparse_features = cov_pop.loc[:, unique_values <= 2].columns.tolist()
    dense_features = cov_pop.loc[:, unique_values > 2].columns.tolist()
    return dense_features, sparse_features


class FeatureExtractor:
    def __init__(self, app_dir, train_conf, client_conf):
        self.logger = logger
        # db_conf, client_conf, training_conf for each hospital
        self.app_dir = Path(app_dir)
        self.db_conf = client_conf["db_conf"]
        self.train_conf = train_conf
        self.project = train_conf.project
        self.data_path = Path(client_conf["data_path"])

        self.project_folder = app_dir / "custom" / self.project
        self.data_project_folder = self.data_path / self.project
        os.makedirs(self.data_project_folder, exist_ok=True)

        self.output_folder = self.project_folder / f"{self.project}Results"
        os.makedirs(self.project_folder, exist_ok=True)
        self.exclude_cols = [
            "rowId",
            "subjectId",
            "cohortStartDate",
            "cohortId",
            "ageYear",
            "gender",
            "outcomeCount",
            "timeAtRisk",
            "survivalTime",
            "indexes",
        ]

    def extraction(self, plp_build=0, package_build=1, feature_extraction=1):
        # sourcery skip: raise-specific-error

        _ = self.package_build(
            plp_build, package_build
        )  # R package build - PLP and project package

        if feature_extraction == 1:
            self.renew_run_script(self.db_conf)  #

            attempts = 0
            while attempts < 3:
                try:
                    os_return = os.system(
                        f"Rscript {self.project_folder/ 'temp_run.R'}"
                    )
                    if os_return == 0:
                        break
                    else:
                        raise Exception("Rscript error")
                except Exception:
                    time.sleep(10)
                    attempts += 1

            logger.info("casc")
        return pd.read_parquet(self.data_project_folder / "cov_pop.parquet")

    def split_patients(self, cov_pop):
        cov_pop.indexes = cov_pop.indexes.astype(int)
        test_dt = cov_pop.query("indexes== -1")
        if cov_pop.indexes.nunique() >= 3:
            valid_dt = cov_pop.query("indexes== 1")
            train_dt = cov_pop.query("indexes > 1 ")
        elif cov_pop.indexes.nunique() == 2:
            train_dt = cov_pop.query("indexes== 1")
            train_pats = train_dt.subjectId.unique()
            valid_pats = np.random.choice(
                train_pats, size=len(train_pats) // 5, replace=False
            )

            train_dt = train_dt[~train_dt.subjectId.isin(valid_pats)]
            valid_dt = train_dt[train_dt.subjectId.isin(valid_pats)]
        train_dt = train_dt.reset_index(drop=True)
        valid_dt = valid_dt.reset_index(drop=True)
        test_dt = test_dt.reset_index(drop=True)
        return train_dt, valid_dt, test_dt

    def renew_run_script(self, db_conf=None):

        assert db_conf is not None, "db_conf is None"
        for key in db_conf.keys():
            assert key is not None, f"{key} is None"

        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)
            logger.info(f"Old {self.output_folder} is removed")

        run_script = f"""
        library({self.project})
        library(dplyr)
        # USER INPUTS
        #=======================
        # The folder where the study intermediate and result files will be written:
        outputFolder <- "{self.output_folder}"

        # Specify where the temporary files (used by the ff package) will be created:
        options(fftempdir = "./temp")

        # Details for connecting to the server:
        dbms <- "{db_conf.dbms}"
        user <- '{db_conf.user}'
        pw <- '{db_conf.pw}'
        server <- '{db_conf.server}'
        port <- '{db_conf.port}'

        downloadJdbcDrivers('{db_conf.dbms}','.')
        connectionDetails <- DatabaseConnector::createConnectionDetails(dbms = dbms,
                                                                        server = server,
                                                                        user = user,
                                                                        password = pw,
                                                                        port = port,
                                                                        pathToDriver='.')

        # Add the database containing the OMOP CDM data
        cdmDatabaseSchema <- '{db_conf.cdmDatabaseSchema}'
        # Add a sharebale name for the database containing the OMOP CDM data
        cdmDatabaseName <- '{db_conf.cdmDatabaseName}'
        # Add a database with read/write access as this is where the cohorts will be generated
        cohortDatabaseSchema <- '{db_conf.cohortDatabaseSchema}'

        oracleTempSchema <- NULL

        # table name where the cohorts will be generated
        cohortTable <- '{self.project}Cohort'
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
                runAnalyses = T,
                createResultsDoc = F,
                packageResults = F,
                createValidationPackage = F,  
                #analysesToValidate = 1,
                minCellCount= 5,
                createShiny = F,
                createJournalDocument = F,
                analysisIdDocument = 1)

        # save cov_pop to parquet
        analysisPath <- file.path(outputFolder,"/Analysis_1")
        CovariateMatrixData <- readRDS(file.path(analysisPath,'CovariateMatrixData.rds'))
        population <- readRDS(file.path(analysisPath,'population.rds'))
        df <- as.data.frame(as.matrix(CovariateMatrixData$data))
        colnames(df) <- CovariateMatrixData$covariateRef$covariateName
        df["rowId"] <- rownames(df)
        pop <- population[,c("rowId", "subjectId", "cohortStartDate", "cohortId", "ageYear", "gender", "outcomeCount", "timeAtRisk", "survivalTime", "indexes")]
        pop_df <- merge(pop, df, by = "rowId")
        
        arrow::write_parquet(pop_df, sink = file.path('{self.data_project_folder}', "cov_pop.parquet"))
        
        # save varImp to parquet

        """
        with open(self.project_folder / "temp_run.R", "w") as f:
            f.write(run_script)
        logger.info("Rscript setted")

    @retry(10, timesleep=10)
    def package_build(self, plp_build=0, project_build=1):
        # PatientLevelPrediction package building
        if plp_build == 1:
            logger.warning(
                """PatientLevelPrediction Package will be replaced by 4.3.7 version"""
            )
            plp_pack_path = self.app_dir / "custom" / "PatientLevelPrediction"

            build_command = (
                f"R CMD INSTALL --no-multiarch --with-keep.source {plp_pack_path}"
            )
            os_return = os.system(build_command)
            if os_return != 0:
                raise Exception("PatientLevelPrediction Package build failed")

            logger.info("PatientLevelPrediction package building done")
        if project_build == 1:
            logger.warning("""Project Package will be build""")
            # 프로젝트 build
            project_pack_path = self.app_dir / "custom" / self.project
            build_command = (
                f"R CMD INSTALL --no-multiarch --with-keep.source {project_pack_path}"
            )
            os_return = os.system(build_command)
            if os_return != 0:
                raise Exception("PatientLevelPrediction Package build failed")
            logger.info(f"{self.project} package building done")
        return "build done"


class CtrFeatureExtractor(FeatureExtractor):
    def __init__(self, app_dir, train_conf, client_conf):
        super().__init__(app_dir, train_conf, client_conf)

    def format_to_ctr(self, cov_pop, selected_features=None):
        # feature_selection 들어가있으면...

        cov_pop.columns = cov_pop.columns.str.replace(
            ".", " ", regex=True
        )  # pytorch does not permit module name with "."

        for col in [col for col in cov_pop.columns if col not in self.exclude_cols]:
            if int(cov_pop[col].sum()) == len(cov_pop[col]):
                cov_pop[col] = 0.0
        # substitute all 1.0 columns to 0.0 for DeepCTR embedding

        train_dt, valid_dt, test_dt = split_patients(cov_pop)
        dense_features, sparse_features = get_dense_sparse_features(cov_pop)
        if len(dense_features) != 0:
            mms = MinMaxScaler(feature_range=(0, 1))
            train_dt, valid_dt, test_dt = self.scaling(
                train_dt, valid_dt, test_dt, dense_features, mms
            )

        fixlen_feature_columns = [
            SparseFeat(feat, 2)
            # TODO : 현재 categorical은 binary 변수밖에 못넣음
            for feat in sparse_features
        ] + [
            DenseFeat(
                feat,
                1,
            )
            for feat in dense_features
        ]
        dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns

        if selected_features is not None:  # selected features 정해져있으면...
            dnn_feature_columns = [
                i for i in dnn_feature_columns if i.name() in selected_features
            ]
            linear_feature_columns = [
                i for i in linear_feature_columns if i.name() in selected_features
            ]

        feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
        train_label = np.array(train_dt["outcomeCount"])
        valid_label = np.array(valid_dt["outcomeCount"])
        test_label = np.array(test_dt["outcomeCount"])
        train_model_input = {name: train_dt[name] for name in feature_names}
        valid_model_input = {name: valid_dt[name] for name in feature_names}
        test_model_input = {name: test_dt[name] for name in feature_names}

        key_with_all_plus = [
            key
            for key in train_model_input
            if (int(train_model_input[key].sum()) == len(train_model_input[key]))
            | (int(valid_model_input[key].sum()) == len(valid_model_input[key]))
            | (int(test_model_input[key].sum()) == len(test_model_input[key]))
        ]
        for key in key_with_all_plus:
            train_model_input[key] = pd.Series(np.zeros_like(train_model_input[key]))
            valid_model_input[key] = pd.Series(np.zeros_like(valid_model_input[key]))
            test_model_input[key] = pd.Series(np.zeros_like(test_model_input[key]))

        return (
            train_model_input,
            valid_model_input,
            test_model_input,
            train_label,
            valid_label,
            test_label,
            linear_feature_columns,
            dnn_feature_columns,
        )

    def get_dense_sparse_features(self, cov_pop):

        cov_pop_ex = cov_pop.drop(self.exclude_cols, axis=1)
        unique_values = cov_pop_ex.nunique()
        sparse_features = cov_pop_ex.loc[:, unique_values <= 2].columns.tolist()
        dense_features = cov_pop_ex.loc[:, unique_values > 2].columns.tolist()
        return dense_features, sparse_features

    def scaling(self, train_dt, valid_dt, test_dt, dense_features, mms):
        train_dt[dense_features] = mms.fit_transform(train_dt[dense_features])
        valid_dt[dense_features] = mms.transform(valid_dt[dense_features])
        test_dt[dense_features] = mms.transform(test_dt[dense_features])
        return train_dt, valid_dt, test_dt
