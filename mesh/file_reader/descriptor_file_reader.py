import re
from mesh.record.mesh_descriptor_record import MeshDescriptorRecord
import typing as tp
from mesh.file_reader.ascii_file_reader import AscIIFileReader


class DescriptorAscIIFileReader(AscIIFileReader):
    def __init__(self, file_path: str):
        """
        Descriptor ASCII file is named as d<four digit year>.bin.
        FIELDS, for details see https://www.nlm.nih.gov/mesh/xml_data_elements.html; extracted fields are:
            *NEWRECORD: INDICATE THE START OF A NEW RECORD
            RECTYPE: RECORD TYPE, THIS SHOULD BE "D" for a descriptor file
            MH: the MeSH Heading
            DC: The class type of this record see below
            ENTRY: related concepts
            PRINT ENTRY: related concepts

        The following documentation are modified from the MeSH Record Types documents.

        This record type plays a central role in MeSH vocabulary as a unit of Indexing and retrieval. With the exception of Class 3 Descriptors, all descriptors are organised
        into a numbered tree structure or hierarchy that allows users to browse in a orderly fashion from broader to narrower topics. Descriptors are divided into four classes.

        Class 1 Descriptors - Main Headings These records are topical headings that are used to index citations in NLM's MEDLINE database, for cataloging of publications,
        and other databases, and are searchable in PubMed as [MH]. Most Descriptors indicate the subject of an indexed string, such as a journal article, that is,
        what the article is about. Descriptors are generally updated on an annual base but may, on occasion, be updated more frequently.

        Class 2 Descriptors - Publication Characteristics (Publication Types) These records indicate what the indexed string is, i.e., its genre, rather than what it is about,
        for example, Historical Article. They may include Publication Components, such as Charts; Publication Formats, such as Editorial; and Study Characteristics,
        such as Clinical Trial. They function as metadata, rather than being about the content. These records are searchable in PubMed as Publication Type [PT],
        and the terms in MEDLINE records are labeled as "PT" or <PublicationType> rather than "MH" or <MeSHHeading>. They are listed in category V of the MeSH Tree Structures. A
        list is available of Publication Types, with Scope Notes.

        Class 3 Descriptors - Check Tags This class of descriptors is used solely for tagging citations that contain certain categories of information. They do not appear in the
        MeSH tree. Modernization has largely eliminated the need for the data type and many of the Check Tags have been changed to Class 1 headings that can be used either a MH
        or a Check Tag. Currently only two Class 3 descriptors remain: "Male" and "Female".

        Class 4 Descriptors - Geographics Descriptors which include continents, regions, countries, states, and other geographic subdivisions. They are not used to characterize
        subject content but rather physical location. They are listed in category Z of the MeSH Tree Structures.


        :param file_path:
        """
        self._mesh_term_pattern: tp.Pattern = re.compile(r'^MH = (.+)$')
        self._mesh_entry_pattern: tp.Pattern = re.compile(r'^(?:PRINT )?ENTRY = ([^|]+).*$')
        self._mesh_number_pattern: tp.Pattern = re.compile(r'^MN = (.+)$')
        self._new_record_pattern = "*NEWRECORD\n"
        super(DescriptorAscIIFileReader, self).__init__(file_path)

    def __next__(self):
        heading = None
        entries = list()
        numbers = list()

        # find the start of record
        while self._line != self._new_record_pattern:
            self._line = self._file_handler.readline()
            if self._line == '':
                # end of file
                break

        # find the heading of the record
        while self._line != '':
            mesh_heading_match = self._mesh_term_pattern.match(self._line)
            if mesh_heading_match:
                # found
                heading = mesh_heading_match.group(1).strip().lower()
                break

            self._line = self._file_handler.readline()

        if heading:
            while self._line != '':
                mesh_entry_match = self._mesh_entry_pattern.match(self._line)
                if mesh_entry_match:
                    # found entries
                    entries.append(mesh_entry_match.group(1).strip().lower())
                else:
                    mesh_number_re = self._mesh_number_pattern.match(self._line)
                    if mesh_number_re:
                        # found numbers
                        numbers.append(mesh_number_re.group(1).strip().lower())
                    else:
                        # check ending of the record
                        if self._line == self._new_record_pattern:
                            break

                self._line = self._file_handler.readline()

            return MeshDescriptorRecord(heading, entries, numbers)

        raise StopIteration
