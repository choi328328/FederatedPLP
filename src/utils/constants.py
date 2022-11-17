'''
get_data에 필요한 constants들을 정의

'''

from dataclasses import dataclass

@dataclass
class PLPConstants:
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
    basics_outcome=basics + ['outcomeCount']