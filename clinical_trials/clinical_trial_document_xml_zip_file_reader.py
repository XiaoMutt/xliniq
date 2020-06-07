from xml.etree import ElementTree
from clinical_trials.clinical_trial_document import ClinicalTrialDocument
from base.file_reader import FileReader
from zipfile import ZipFile


class ClinicalTrialDocumentXmlZipFileReader(FileReader):
    _parse_routes = {
        'nct_id': ('id_info', 'nct_id'),
        'brief_title': ('brief_title',),
        'official_title': ('official_title',),
        'brief_summary': ('brief_summary', 'textblock'),
        'detailed_description': ('detailed_description', 'textblock'),
        'study_first_submitted': ('study_first_submitted',),
        'condition': ('condition',),
        'eligibility': ('eligibility', 'criteria', 'textblock')
    }

    def __init__(self, file_path):
        super(ClinicalTrialDocumentXmlZipFileReader, self).__init__(file_path)
        self._file_name_itr = None

    def __enter__(self):
        if self._file_handler is None:
            self._file_handler = ZipFile(self._file_path, 'r')
        self._file_name_itr = self._file_handler.namelist().__iter__()

        return self

    def __iter__(self):
        return self.__enter__()

    def __next__(self):
        file_name = next(self._file_name_itr)
        while not file_name.endswith('xml'):
            file_name = next(self._file_name_itr)
        xml = self._file_handler.open(file_name).read().decode()
        return self._parse(xml)

    def _parse(self, xml: str):
        tree = ElementTree.fromstring(xml)
        record = dict()
        for key, route in self._parse_routes.items():
            current = tree
            for el in route:
                current = current.find(el)
                if current is None:
                    break
            record[key] = None if current is None else current.text

        return ClinicalTrialDocument(**record)
