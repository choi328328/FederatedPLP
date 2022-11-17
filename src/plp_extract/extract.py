
 
from loguru import logger
import os
from pathlib import Path
from .make_rscript import renew_run_script

'''
PLP package를 입력으로 받아서 

1) PLP package build
2) temporary R script 생성
3) R script 실행해서 feature extraction to temporary folder (as parquet)
'''

def extract(train_conf, client_conf):
    '''
    PLP package를 입력으로 받아서 

    1) PLP package build
    2) temporary R script 생성
    3) R script 실행해서 feature extraction to temporary folder (as parquet)
    '''
    package=train_conf.package
    project= train_conf.project
    
    os.makedirs('./temps', exist_ok=True)
    build_plp(package)  # R package build - PLP and project package
    run_script=renew_run_script(train_conf=train_conf, client_conf=client_conf)  # renew run script -> ./temps/temp_run.R"
    with open(f"./temps/{project}_run.R", "w") as f:
        f.write(run_script)
    
    os.system(f"Rscript ./temps/{project}_run.R") # run R script -> extract features as parquet to temp folder 

def build_plp(package):
    '''
    PLP package build
    '''
    logger.info(f"""{package} package will be build""")
    project_pack_path = Path('./packages') / package
    build_command = (
    f"R CMD INSTALL --no-multiarch --with-keep.source {project_pack_path}"
    )
    os_return = os.system(build_command)
    if os_return != 0:
        raise Exception("PatientLevelPrediction Package build failed")
    logger.info(f"{package} package building done")


