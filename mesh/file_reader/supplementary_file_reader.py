from mesh.file_reader.ascii_file_reader import AscIIFileReader
from mesh.record.mesh_descriptor_record import MeshDescriptorRecord
import typing as tp
import re


class SupplementaryRecordAscIIFileReader(AscIIFileReader):
    def __init__(self, file_name: str):
        """
        Supplementary Record ASCII file is named as c<four digit year>.bin.
        FIELDS, for details see https://www.nlm.nih.gov/mesh/xml_data_elements.html; extracted fields are:
            *NEWRECORD: INDICATE THE START OF A NEW RECORD
            RECTYPE: RECORD TYPE, THIS SHOULD BE "C" for a supplementary record file
            MH: the MeSH Heading
            HM: HEADING MAPPED-TO.
            SY: Alpha-numeric string which comprises the basic unit of the MeSH vocabulary. Also functions as the name of a Descriptor and concept
            PI: previous indexing. This field follow the year range in a pair of parentheses

        Supplementary Records, also called Supplementary Chemical Records(SCRs), are used to index chemicals, drugs, and other concepts such as rare diseases for MEDLINE and are
        searchable by Substance Name [NM] in PubMed. Unlike Descriptors, SCRs are not organised in a tree hierarchy. Instead each SCR is linked to one or more Descriptors by the
        Heading Mapped To (HM) field in the SCR. They also include a Indexing Information (II) field field that is used to refer to other descriptors that are from related
        topics. SCRs are created daily and distributed nightly Monday-Thursday. There are currently over 230,000 SCR records with over 505,000 SCR terms. Like all MeSH records,
        SCRs are searchable in the MeSH Browser. Four classes of SCRs exist.

        Class 1 Supplementary Records - Chemicals
        These records are dedicated to chemicals and are primarily heading mapped to the D tree descriptors.

        Class 2 Supplementary Records - Protocols
        These records are dedicated to Chemotherapy Protocols. They are heading mapped to the MeSH heading "Antineoplastic Combined Chemotherapy Protocols" and to chemicals used in the protocols found in D tree descriptors.

        Class 3 Supplementary Records - Diseases
        These records are dedicated to diseases and are primarily heading mapped to the C tree descriptors and anatomical headings found in the A tree.

        Class 4 Supplementary Records - Organisms (new for 2018 MeSH)
        These records are dedicated to organisms (e.g., viruses) and are primarily heading mapped to the B tree organism descriptors.
        :param file_name:
        """
        self.mesh_term_pattern: tp.Pattern = re.compile(r'^MH = (.+)$')
        self.mesh_entry_pattern: tp.Pattern = re.compile(r'^(?:PRINT )?ENTRY = ([^|]+).*$')
        self.mesh_number_pattern: tp.Pattern = re.compile(r'^MN = (.+)$')
        self.new_record_pattern = "*NEWRECORD\n"

        super(SupplementaryRecordAscIIFileReader, self).__init__(file_name)

    def __next__(self):
        heading = None
        entries = list()
        numbers = list()

        line: str = self._file_handler.readline()
        while line and line != self.new_record_pattern:
            line = self._file_handler.readline()

        mesh_heading_match = None
        while line and mesh_heading_match is None:
            # found new record
            mesh_heading_match = self.mesh_term_pattern.match(line)
            if mesh_heading_match:
                heading = mesh_heading_match.group(1).strip().lower()

            line = self._file_handler.readline()

        while line and mesh_heading_match is not None and line != self.new_record_pattern:
            mesh_entry_match = self.mesh_entry_pattern.match(line)
            if mesh_entry_match:
                entries.append(mesh_entry_match.group(1).strip().lower())
            else:
                mesh_number_re = self.mesh_number_pattern.match(line)
                if mesh_number_re:
                    numbers.append(mesh_number_re.group(1).strip().lower())

            line = self._file_handler.readline()

        if heading:
            return MeshDescriptorRecord(heading, entries, numbers)
        else:
            raise StopIteration
