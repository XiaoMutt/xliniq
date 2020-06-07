from base.immutable import Immutable
from datetime import datetime


class ClinicalTrialDocument(Immutable):
    nct_id: str
    brief_title: str
    official_title: str
    brief_summary: str
    detailed_description: str
    study_first_submitted: datetime
    condition: str
    eligibility: str

    MESH_ATTRIBUTES = (
        'brief_title',
        'official_title',
        'brief_summary',
        'detailed_description',
        'condition',
        'eligibility',
    )

    def __init__(self, nct_id: str, brief_title: str, official_title: str,
                 brief_summary: str, detailed_description: str, study_first_submitted: datetime,
                 condition: str, eligibility: str):
        if type(study_first_submitted) == str:
            study_first_submitted = datetime.strptime(study_first_submitted, '%B %d, %Y')

        super(ClinicalTrialDocument, self).__init__(nct_id=nct_id,
                                                    brief_title=brief_title,
                                                    official_title=official_title,
                                                    brief_summary=brief_summary,
                                                    detailed_description=detailed_description,
                                                    study_first_submitted=study_first_submitted,
                                                    condition=condition,
                                                    eligibility=eligibility)
